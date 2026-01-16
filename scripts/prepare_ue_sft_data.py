"""
UnrealEngine SFT数据集构造脚本
==============================

功能：
1. 从UE代码中自动提取类/函数信息，生成QA对
2. 支持手动补充高质量QA对
3. 利用大模型API自动生成QA（可选）

使用方法：
python prepare_ue_sft_data.py --ue_source_path "D:/UnrealEngine/Engine/Source" --output_path "../dataset/ue_sft.jsonl"
"""

import os
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import random


@dataclass
class UEClassInfo:
    """UE类信息"""
    name: str
    parent_class: str
    file_path: str
    description: str
    functions: List[str]
    properties: List[str]
    macros: List[str]  # UCLASS, USTRUCT等
    code_snippet: str


class UECodeParser:
    """UE代码解析器"""
    
    # UE常见宏
    UE_MACROS = {
        'UCLASS': '定义一个UObject派生类',
        'USTRUCT': '定义一个结构体',
        'UENUM': '定义一个枚举',
        'UFUNCTION': '定义一个可被反射的函数',
        'UPROPERTY': '定义一个可被反射的属性',
        'UINTERFACE': '定义一个接口',
        'GENERATED_BODY': '生成反射代码',
        'GENERATED_UCLASS_BODY': '生成UClass反射代码',
    }
    
    # 常见UE基类
    UE_BASE_CLASSES = {
        'UObject': '所有UE对象的基类',
        'AActor': '可放置在世界中的对象基类',
        'UActorComponent': '组件基类',
        'APawn': '可被控制的Actor基类',
        'ACharacter': '角色基类，带移动组件',
        'APlayerController': '玩家控制器基类',
        'UGameInstance': '游戏实例基类',
        'UWorld': '世界对象',
        'ULevel': '关卡对象',
        'UWidget': 'UI控件基类',
        'UUserWidget': '用户自定义UI基类',
    }
    
    def __init__(self, ue_source_path: str):
        self.ue_source_path = Path(ue_source_path)
        
    def parse_header_file(self, file_path: Path) -> List[UEClassInfo]:
        """解析头文件，提取类信息"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            return []
        
        classes = []
        
        # 匹配UCLASS定义
        uclass_pattern = r'UCLASS\(([^)]*)\)\s*class\s+(?:\w+_API\s+)?(\w+)\s*(?::\s*public\s+(\w+))?'
        
        for match in re.finditer(uclass_pattern, content):
            macros = match.group(1)
            class_name = match.group(2)
            parent_class = match.group(3) or 'UObject'
            
            # 提取类的代码片段
            start_pos = match.start()
            # 找到类的结束位置（简化处理：找下一个UCLASS或文件结尾）
            next_match = re.search(r'\n(?:UCLASS|USTRUCT|$)', content[match.end():])
            end_pos = match.end() + next_match.start() if next_match else len(content)
            
            class_code = content[start_pos:end_pos]
            
            # 提取函数
            functions = self._extract_functions(class_code)
            
            # 提取属性
            properties = self._extract_properties(class_code)
            
            # 生成描述
            description = self._extract_comment(content, start_pos)
            
            try:
                rel_path = file_path.relative_to(self.ue_source_path)
            except ValueError:
                rel_path = file_path.name
            
            classes.append(UEClassInfo(
                name=class_name,
                parent_class=parent_class,
                file_path=str(rel_path),
                description=description,
                functions=functions[:10],  # 限制数量
                properties=properties[:10],
                macros=[macros] if macros else [],
                code_snippet=class_code[:2000]  # 限制长度
            ))
        
        return classes
    
    def _extract_functions(self, code: str) -> List[str]:
        """提取函数签名"""
        # 匹配UFUNCTION修饰的函数
        pattern = r'UFUNCTION\([^)]*\)\s*(?:virtual\s+)?(\w+(?:\s*[*&])?\s+\w+\s*\([^)]*\))'
        matches = re.findall(pattern, code)
        
        # 也匹配普通virtual函数
        pattern2 = r'virtual\s+(\w+(?:\s*[*&])?\s+\w+\s*\([^)]*\))\s*(?:override|const)?;'
        matches2 = re.findall(pattern2, code)
        
        return list(set(matches + matches2))
    
    def _extract_properties(self, code: str) -> List[str]:
        """提取属性"""
        pattern = r'UPROPERTY\([^)]*\)\s*(\w+(?:\s*[*&<>][\w\s,*&<>]*)?\s+\w+);'
        return re.findall(pattern, code)
    
    def _extract_comment(self, content: str, pos: int) -> str:
        """提取类前的注释"""
        # 向前查找注释
        before = content[max(0, pos-500):pos]
        
        # 匹配多行注释
        multi_comment = re.findall(r'/\*\*(.*?)\*/', before, re.DOTALL)
        if multi_comment:
            comment = multi_comment[-1]
            # 清理注释格式
            comment = re.sub(r'\s*\*\s*', ' ', comment)
            return comment.strip()[:200]
        
        # 匹配单行注释
        single_comments = re.findall(r'//\s*(.+)', before)
        if single_comments:
            return single_comments[-1].strip()[:200]
        
        return ""


class UESFTDataGenerator:
    """UE SFT数据生成器"""
    
    # 问题模板
    QUESTION_TEMPLATES = [
        # 类相关
        ("什么是{class_name}类？", "class_intro"),
        ("{class_name}类的作用是什么？", "class_intro"),
        ("{class_name}继承自哪个类？", "inheritance"),
        ("如何使用{class_name}？", "usage"),
        ("{class_name}有哪些主要函数？", "functions"),
        ("{class_name}有哪些属性？", "properties"),
        ("{class_name}在哪个头文件中定义？", "file_location"),
        
        # 功能相关
        ("UE中如何实现{feature}？", "feature"),
        ("在Unreal Engine中，{feature}应该用什么类？", "feature"),
        
        # 代码相关
        ("请提供{class_name}的代码示例", "code_example"),
        ("{class_name}的基本用法代码是什么？", "code_example"),
    ]
    
    # 功能到类的映射
    FEATURE_CLASS_MAP = {
        "角色移动": ["ACharacter", "UCharacterMovementComponent"],
        "动画播放": ["UAnimInstance", "USkeletalMeshComponent"],
        "物理碰撞": ["UPrimitiveComponent", "FHitResult"],
        "UI界面": ["UUserWidget", "UWidgetComponent"],
        "音频播放": ["UAudioComponent", "USoundBase"],
        "粒子特效": ["UParticleSystemComponent", "UNiagaraComponent"],
        "AI行为树": ["UBehaviorTree", "UAIController"],
        "导航寻路": ["UNavigationSystemV1", "ANavigationData"],
        "输入处理": ["UInputComponent", "APlayerController"],
        "网络复制": ["AActor", "UActorComponent"],
        "保存游戏": ["USaveGame", "UGameplayStatics"],
        "定时器": ["FTimerHandle", "UWorld"],
        "材质参数": ["UMaterialInstanceDynamic", "UMaterialInterface"],
        "光照": ["ULightComponent", "UDirectionalLightComponent"],
        "相机": ["UCameraComponent", "APlayerCameraManager"],
    }
    
    def __init__(self, parser: UECodeParser):
        self.parser = parser
        self.class_infos: Dict[str, UEClassInfo] = {}
        
    def collect_class_info(self, max_files: int = None):
        """收集所有类信息"""
        header_files = []
        
        for root, dirs, files in os.walk(self.parser.ue_source_path):
            # 跳过不需要的目录
            dirs[:] = [d for d in dirs if d not in {'ThirdParty', 'Intermediate', 'Binaries'}]
            
            for f in files:
                if f.endswith('.h'):
                    header_files.append(Path(root) / f)
        
        if max_files:
            header_files = header_files[:max_files]
        
        print(f"解析 {len(header_files)} 个头文件...")
        
        for file_path in tqdm(header_files, desc="解析头文件"):
            classes = self.parser.parse_header_file(file_path)
            for cls in classes:
                self.class_infos[cls.name] = cls
        
        print(f"共收集 {len(self.class_infos)} 个类定义")
        
    def generate_qa_pairs(self) -> List[Dict]:
        """生成QA对"""
        qa_pairs = []
        
        # 1. 基于类信息生成QA
        for class_name, info in tqdm(self.class_infos.items(), desc="生成QA对"):
            qa_pairs.extend(self._generate_class_qa(info))
        
        # 2. 基于功能生成QA
        qa_pairs.extend(self._generate_feature_qa())
        
        # 3. 添加通用UE知识QA
        qa_pairs.extend(self._generate_general_qa())
        
        return qa_pairs
    
    def _generate_class_qa(self, info: UEClassInfo) -> List[Dict]:
        """基于类信息生成QA"""
        qa_list = []
        
        # 类介绍
        if info.description:
            qa_list.append(self._create_conversation(
                f"什么是{info.name}类？",
                f"{info.name}是一个Unreal Engine类。\n\n"
                f"**描述**: {info.description}\n"
                f"**父类**: {info.parent_class}\n"
                f"**头文件**: {info.file_path}"
            ))
        
        # 继承关系
        qa_list.append(self._create_conversation(
            f"{info.name}类继承自哪个类？",
            f"{info.name}类继承自 **{info.parent_class}**。\n\n"
            f"头文件位置: `{info.file_path}`"
        ))
        
        # 文件位置
        qa_list.append(self._create_conversation(
            f"{info.name}定义在哪个文件中？",
            f"**{info.name}** 定义在以下文件中：\n\n"
            f"```\n{info.file_path}\n```"
        ))
        
        # 函数列表
        if info.functions:
            functions_str = "\n".join([f"- `{f}`" for f in info.functions[:5]])
            qa_list.append(self._create_conversation(
                f"{info.name}有哪些主要函数？",
                f"**{info.name}** 的主要函数包括：\n\n{functions_str}\n\n"
                f"更多详情请查看: `{info.file_path}`"
            ))
        
        # 属性列表
        if info.properties:
            props_str = "\n".join([f"- `{p}`" for p in info.properties[:5]])
            qa_list.append(self._create_conversation(
                f"{info.name}有哪些属性？",
                f"**{info.name}** 的主要属性包括：\n\n{props_str}\n\n"
                f"这些属性使用UPROPERTY宏修饰，支持蓝图访问和反射。"
            ))
        
        # 代码示例
        if len(info.code_snippet) > 100:
            qa_list.append(self._create_conversation(
                f"请提供{info.name}的代码定义",
                f"**{info.name}** 的代码定义如下：\n\n"
                f"```cpp\n{info.code_snippet[:1500]}\n```\n\n"
                f"完整代码请查看: `{info.file_path}`"
            ))
        
        return qa_list
    
    def _generate_feature_qa(self) -> List[Dict]:
        """基于功能生成QA"""
        qa_list = []
        
        for feature, classes in self.FEATURE_CLASS_MAP.items():
            # 查找相关类的信息
            related_infos = [self.class_infos.get(c) for c in classes if c in self.class_infos]
            
            if related_infos:
                classes_str = ", ".join([f"**{c}**" for c in classes])
                details = []
                for info in related_infos:
                    if info:
                        details.append(f"- **{info.name}**: 位于 `{info.file_path}`")
                
                details_str = "\n".join(details) if details else "请查阅UE官方文档获取更多信息。"
                
                qa_list.append(self._create_conversation(
                    f"UE中如何实现{feature}？",
                    f"在Unreal Engine中实现**{feature}**，主要使用以下类：\n\n"
                    f"{classes_str}\n\n"
                    f"**相关文件**:\n{details_str}"
                ))
        
        return qa_list
    
    def _generate_general_qa(self) -> List[Dict]:
        """生成通用UE知识QA"""
        general_qa = [
            {
                "q": "UE的反射系统是什么？",
                "a": "Unreal Engine的反射系统是通过一系列宏实现的元数据系统：\n\n"
                     "**核心宏**:\n"
                     "- `UCLASS()`: 标记类可被反射\n"
                     "- `UPROPERTY()`: 标记属性可被反射\n"
                     "- `UFUNCTION()`: 标记函数可被反射\n"
                     "- `USTRUCT()`: 标记结构体可被反射\n"
                     "- `UENUM()`: 标记枚举可被反射\n\n"
                     "这些宏由Unreal Header Tool (UHT)处理，生成反射代码。"
            },
            {
                "q": "AActor和UActorComponent有什么区别？",
                "a": "**AActor**:\n"
                     "- 可以放置在世界中的对象\n"
                     "- 有Transform（位置、旋转、缩放）\n"
                     "- 可以被Spawn和Destroy\n"
                     "- 文件: `Engine/Classes/GameFramework/Actor.h`\n\n"
                     "**UActorComponent**:\n"
                     "- 附加到Actor上的功能模块\n"
                     "- 不能独立存在于世界中\n"
                     "- 提供可复用的功能（移动、渲染等）\n"
                     "- 文件: `Engine/Classes/Components/ActorComponent.h`"
            },
            {
                "q": "UE中的Tick函数是什么？",
                "a": "**Tick函数**是每帧调用的更新函数：\n\n"
                     "**AActor::Tick(float DeltaTime)**\n"
                     "- Actor的每帧更新\n"
                     "- 需要设置 `PrimaryActorTick.bCanEverTick = true`\n\n"
                     "**UActorComponent::TickComponent(float DeltaTime, ...)**\n"
                     "- 组件的每帧更新\n"
                     "- 需要设置 `PrimaryComponentTick.bCanEverTick = true`\n\n"
                     "```cpp\n"
                     "void AMyActor::Tick(float DeltaTime)\n"
                     "{\n"
                     "    Super::Tick(DeltaTime);\n"
                     "    // 每帧逻辑\n"
                     "}\n"
                     "```"
            },
            {
                "q": "如何在UE中创建定时器？",
                "a": "使用 **FTimerHandle** 和 **GetWorldTimerManager()**:\n\n"
                     "```cpp\n"
                     "// 头文件中声明\n"
                     "FTimerHandle MyTimerHandle;\n\n"
                     "// 设置定时器\n"
                     "GetWorldTimerManager().SetTimer(\n"
                     "    MyTimerHandle,\n"
                     "    this,\n"
                     "    &AMyActor::MyFunction,\n"
                     "    1.0f,  // 间隔时间\n"
                     "    true   // 是否循环\n"
                     ");\n\n"
                     "// 清除定时器\n"
                     "GetWorldTimerManager().ClearTimer(MyTimerHandle);\n"
                     "```\n\n"
                     "**相关类**: FTimerHandle, FTimerManager\n"
                     "**头文件**: `Engine/Public/TimerManager.h`"
            },
            {
                "q": "UE中UPROPERTY有哪些常用说明符？",
                "a": "**UPROPERTY常用说明符**:\n\n"
                     "**可见性**:\n"
                     "- `VisibleAnywhere`: 所有地方可见\n"
                     "- `VisibleDefaultsOnly`: 仅类默认值可见\n"
                     "- `VisibleInstanceOnly`: 仅实例可见\n\n"
                     "**可编辑性**:\n"
                     "- `EditAnywhere`: 所有地方可编辑\n"
                     "- `EditDefaultsOnly`: 仅类默认值可编辑\n"
                     "- `EditInstanceOnly`: 仅实例可编辑\n\n"
                     "**蓝图**:\n"
                     "- `BlueprintReadOnly`: 蓝图只读\n"
                     "- `BlueprintReadWrite`: 蓝图读写\n\n"
                     "**分类**:\n"
                     "- `Category=\"MyCategory\"`: 编辑器分类\n\n"
                     "```cpp\n"
                     "UPROPERTY(EditAnywhere, BlueprintReadWrite, Category=\"Combat\")\n"
                     "float Health = 100.0f;\n"
                     "```"
            },
        ]
        
        return [self._create_conversation(qa["q"], qa["a"]) for qa in general_qa]
    
    def _create_conversation(self, question: str, answer: str) -> Dict:
        """创建对话格式"""
        return {
            "conversations": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }
    
    def save_dataset(self, qa_pairs: List[Dict], output_path: str):
        """保存数据集"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 打乱顺序
        random.shuffle(qa_pairs)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')
        
        print(f"\n保存完成！共 {len(qa_pairs)} 条QA对")
        print(f"保存至: {output_path}")


def create_manual_qa_template(output_path: str):
    """创建手动QA模板文件"""
    template = [
        {
            "conversations": [
                {"role": "user", "content": "如何在UE中创建一个新的Actor类？"},
                {"role": "assistant", "content": "在UE中创建新Actor类的步骤：\n\n1. 在编辑器中：File -> New C++ Class\n2. 选择Actor作为父类\n3. 命名你的类（如AMyActor）\n\n**代码结构**:\n```cpp\n// MyActor.h\n#pragma once\n#include \"CoreMinimal.h\"\n#include \"GameFramework/Actor.h\"\n#include \"MyActor.generated.h\"\n\nUCLASS()\nclass MYPROJECT_API AMyActor : public AActor\n{\n    GENERATED_BODY()\npublic:\n    AMyActor();\n    virtual void BeginPlay() override;\n    virtual void Tick(float DeltaTime) override;\n};\n```\n\n**关键文件**:\n- `Engine/Source/Runtime/Engine/Classes/GameFramework/Actor.h`"}
            ]
        },
        {
            "conversations": [
                {"role": "user", "content": "请在这里添加更多QA对..."},
                {"role": "assistant", "content": "按照上面的格式添加更多UE相关的问答对"}
            ]
        }
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for qa in template:
            f.write(json.dumps(qa, ensure_ascii=False, indent=2) + '\n')
    
    print(f"模板文件已创建: {output_path}")
    print("请编辑此文件添加更多高质量QA对")


def main():
    parser = argparse.ArgumentParser(description="UE源代码SFT数据集生成")
    parser.add_argument('--ue_source_path', type=str, required=True,
                        help="UE源代码目录路径")
    parser.add_argument('--output_path', type=str, default='../dataset/ue_sft.jsonl',
                        help="输出文件路径")
    parser.add_argument('--max_files', type=int, default=None,
                        help="最大处理文件数（用于测试）")
    parser.add_argument('--create_template', action='store_true',
                        help="仅创建手动QA模板")
    
    args = parser.parse_args()
    
    if args.create_template:
        create_manual_qa_template(args.output_path.replace('.jsonl', '_template.jsonl'))
        return
    
    if not os.path.exists(args.ue_source_path):
        print(f"错误: 路径不存在 - {args.ue_source_path}")
        return
    
    # 解析代码
    code_parser = UECodeParser(args.ue_source_path)
    generator = UESFTDataGenerator(code_parser)
    
    # 收集类信息
    generator.collect_class_info(max_files=args.max_files)
    
    # 生成QA对
    qa_pairs = generator.generate_qa_pairs()
    
    # 保存
    generator.save_dataset(qa_pairs, args.output_path)


if __name__ == '__main__':
    main()
