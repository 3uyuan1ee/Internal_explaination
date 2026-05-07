# 实验设计方案：概念瓶颈模型 (Concept Bottleneck Model) — 内在可解释性实验

## Context

本实验是 BIT《可解释人工智能》课程报告的一部分，核心目标是：
- 实现一个**内在可解释模型**（Concept Bottleneck Model），让模型的推理过程通过人类可理解的概念透明化
- 与**事后归因方法**（Grad-CAM）进行系统对比，展示内在可解释性 vs 事后可解释性的权衡
- 通过量化评估（准确率损失、概念纯度、干预实验）深入讨论"什么是好的解释"

---

## 一、实验架构总览

```
                    ┌─────────────────────────────────────────────┐
                    │           Concept Bottleneck Model           │
                    │                                              │
  Input Image ──►  │  ResNet-18    ┌──────────┐    ┌───────────┐ │  ──► Class Prediction
  (224×224)        │  Encoder ──►  │ Concepts │──►│ Sparse    │ │      + Concept Explanation
                    │  (frozen)     │ (312-dim) │    │ Linear    │ │
                    │               │ Sigmoid   │    │ Classifier│ │
                    └───────────────┴──────────┘────┴───────────┘ ┘
                                     │
                                     ▼
                              Top-3 Concepts
                        "black cap, gray wing, forked tail"
```

---

## 二、项目文件结构

```
Internal_explaination/
├── config.py                 # 全局配置（路径、超参数、类别选择）
├── data/
│   └── CUB_200_2011/         # 数据集（需手动下载）
├── dataset.py                # CUB数据集加载器（图像+属性+标签）
├── models/
│   ├── baseline.py           # ResNet-18 黑盒基线模型
│   ├── concept_predictor.py  # 概念预测器（ResNet-18 → 312维概念）
│   ├── label_predictor.py    # 标签预测器（概念 → 类别，稀疏线性层）
│   └── cbm.py                # 完整CBM模型（整合概念+标签预测器）
├── train_concept.py          # 阶段1：训练概念预测器
├── train_label.py            # 阶段2：训练标签预测器
├── train_baseline.py         # 训练黑盒基线模型
├── evaluate.py               # 量化评估（准确率、概念纯度、干预实验）
├── explain.py                # 生成解释（概念解释 + Grad-CAM热力图）
├── analyze.py                # 可视化分析与报告图表生成
├── docs/
│   └── EXPERIMENT_DESIGN.md  # 本文件（实验设计文档）
└── outputs/                  # 训练产出（权重、日志、图表）
    ├── checkpoints/
    ├── figures/
    └── logs/
```

---

## 三、数据集准备

### 3.1 数据集选取
- **CUB-200-2011**：11,788 张鸟类图像，200 个类别，312 个二值属性标注
- **子集选择**：从 200 个类别中选取 **24 个类别**（平衡训练规模与多样性）

### 3.2 类别选取策略
选择形态差异明显、属性区分度高的类别，覆盖不同科属：

| 类别编号 | 英文名 | 中文名 | 选取理由 |
|---------|--------|--------|---------|
| 16 | Ovenbird | 灶鸟 | 独特冠斑 |
| 17 | Groove-billed Ani | 沟嘴犀鹃 | 奇特喙形 |
| 22 | Sayornis | 霸鹟 | 体型典型 |
| 36 | Northern Flicker | 北扑翅鴷 | 翅膀色斑明显 |
| 47 | American Robin | 旅鸫 | 橙红色胸部 |
| 49 | European Starling | 欧洲椋鸟 | 金属色羽毛 |
| 55 | Purple Finch | 紫朱雀 | 紫红色体羽 |
| 62 | American Goldfinch | 美洲金翅雀 | 鲜黄色体羽 |
| 63 | House Sparrow | 家麻雀 | 棕灰色典型 |
| 68 | Cliff Swallow | 崖燕 | 叉形尾巴 |
| 73 | Blue Jay | 蓝松鸦 | 蓝色翅膀+冠羽 |
| 85 | Northern Mockingbird | 北美模仿鸟 | 灰色体羽+白翼斑 |
| 96 | Northern Cardinal | 北美红雀 | 鲜红色+冠羽 |
| 100 | Brown-headed Cowbird | 褐头牛鹂 | 棕色头+黑色体 |
| 104 | Baltimore Oriole | 巴尔的摩拟黄鹂 | 橙黑色对比 |
| 112 | Great Crested Flycatcher | 大冠蝇鹟 | 黄色腹部 |
| 122 | White-breasted Nuthatch | 白胸鳾 | 灰蓝色背部+白胸 |
| 124 | House Wren | 鹪鹩 | 小型棕色鸟 |
| 134 | Cedar Waxwing | 雪松太平鸟 | 蜡质红翼尖 |
| 161 | Mourning Warbler | 哀莺 | 灰色头罩 |
| 166 | American Redstart | 橙尾鸲莺 | 橙黑斑块 |
| 178 | Blue-headed Vireo | 蓝头绿鹃 | 蓝灰色头+白眉纹 |
| 189 | Pine Siskin | 松金翅雀 | 棕色条纹+黄色翼斑 |
| 196 | House Finch | 家朱雀 | 红色头胸 |

### 3.3 数据预处理
```python
# 图像预处理流水线
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

### 3.4 属性筛选
从 312 个属性中筛选出在选定的 24 个类别中**方差 > 0.05** 的属性（去除全0/全1的无效属性），预计保留 **150~200 个有效概念**。

---

## 四、模型架构详细设计

### 4.1 黑盒基线模型 (Baseline)

```python
# 标准 ResNet-18 直接分类
class BaselineModel(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, num_classes)
```

**训练参数：**
- 优化器：SGD (lr=0.001, momentum=0.9, weight_decay=1e-4)
- 学习率调度：StepLR (step_size=20, gamma=0.1)
- Epochs: 50
- Batch size: 32
- 损失函数：CrossEntropyLoss

### 4.2 概念瓶颈模型 (CBM)

#### 4.2.1 概念预测器
```python
class ConceptPredictor(nn.Module):
    def __init__(self, num_concepts, backbone='resnet18'):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        # 截断到 avgpool 层，输出 512 维特征
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # 去掉 fc
        self.flatten = nn.Flatten()
        # 概念预测头
        self.concept_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_concepts)
        )

    def forward(self, x):
        features = self.flatten(self.encoder(x))  # [B, 512]
        concept_logits = self.concept_head(features)  # [B, num_concepts]
        concept_probs = torch.sigmoid(concept_logits)  # [B, num_concepts]
        return concept_probs
```

**阶段1训练参数：**
- 优化器：Adam (lr=1e-4, weight_decay=1e-4)
- 学习率调度：CosineAnnealingLR
- Epochs: 40
- 损失函数：BCEWithLogitsLoss（逐概念二元交叉熵）

#### 4.2.2 标签预测器（稀疏线性层）
```python
class LabelPredictor(nn.Module):
    def __init__(self, num_concepts, num_classes):
        super().__init__()
        # 关键：使用稀疏线性层，保证可解释性
        self.linear = nn.Linear(num_concepts, num_classes)
        # 可选：添加 L1 正则化促进稀疏性

    def forward(self, concepts):
        return self.linear(concepts)  # [B, num_classes]
```

**阶段2训练参数：**
- 冻结概念预测器权重
- 优化器：Adam (lr=1e-3)
- Epochs: 30
- 损失函数：CrossEntropyLoss + L1 正则化（lambda=0.01，促进权重稀疏）

#### 4.2.3 完整 CBM 模型
```python
class ConceptBottleneckModel(nn.Module):
    def __init__(self, num_concepts, num_classes):
        super().__init__()
        self.concept_predictor = ConceptPredictor(num_concepts)
        self.label_predictor = LabelPredictor(num_concepts, num_classes)

    def forward(self, x, intervene_concepts=None):
        concept_probs = self.concept_predictor(x)
        # 支持概念干预
        if intervene_concepts is not None:
            concept_probs = intervene_concepts
        class_logits = self.label_predictor(concept_probs)
        return concept_probs, class_logits
```

---

## 五、训练流程

### 5.1 两阶段训练

```
阶段1: 训练概念预测器
  输入: 图像 → ResNet-18 编码器 → 概念预测头 → 312维概念概率
  监督: 属性标注 (312个二值标签)
  损失: Σ BCE(ĉ_i, c_i), i=1..312
  目标: 让概念预测器学会从图像中提取人类可理解的概念

阶段2: 训练标签预测器
  输入: 概念概率 (来自冻结的概念预测器) → 稀疏线性层 → 24维类别logits
  监督: 类别标注 (24个类别)
  损失: CrossEntropy + λ·L1(W)
  目标: 学会用概念组合来区分类别，L1正则化保证每个类别只依赖少数关键概念
```

### 5.2 训练监控指标
- 阶段1：概念预测平均准确率、各概念AUC
- 阶段2：分类准确率、权重稀疏度（非零权重占比）

---

## 六、解释生成模块

### 6.1 概念级解释（CBM）
```python
def generate_concept_explanation(model, image, attribute_names, class_names, top_k=3):
    """
    对单张测试图像生成自然语言概念解释。

    流程:
    1. 前向传播得到 concept_probs [num_concepts] 和 class_logits [num_classes]
    2. 取 argmax 得到预测类别
    3. 获取该类别对应的 label_predictor 权重列 W[:, pred_class]
    4. 计算 concept_probs * |W[:, pred_class]| 得到每个概念的重要性分数
    5. 取 top-k 概念，格式化为自然语言
    """
    concept_probs, class_logits = model(image)
    pred_class = class_logits.argmax(dim=1).item()

    # 获取类别对应的概念权重
    class_weights = model.label_predictor.linear.weight[pred_class]  # [num_concepts]
    importance_scores = concept_probs.squeeze() * class_weights.abs()

    # Top-k 概念
    topk_values, topk_indices = importance_scores.topk(top_k)

    # 格式化输出
    explanation = f"预测为 {class_names[pred_class]}，因为：\n"
    for val, idx in zip(topk_values, topk_indices):
        concept_name = attribute_names[idx]
        activation = concept_probs.squeeze()[idx].item()
        explanation += f"  - {concept_name} (激活值: {activation:.2f}, 重要性: {val:.3f})\n"

    return explanation
```

**输出示例：**
```
预测为 Baltimore Oriole，关键概念：
  - has_wing_color_black (激活值: 0.92, 权重: 2.31)
  - has_belly_color_orange (激活值: 0.87, 权重: 1.89)
  - has_tail_pattern_solid (激活值: 0.78, 权重: 1.45)

自然语言：预测为巴尔的摩拟黄鹂，因为它具有黑色的翅膀、橙色的腹部、纯色尾巴。
```

### 6.2 Grad-CAM 热力图（基线对比）
```python
# 使用 pytorch-grad-cam 库
from pytorch_grad_cam import GradCAM

def generate_gradcam_explanation(baseline_model, image, target_class=None):
    """
    对黑盒基线模型生成 Grad-CAM 热力图。
    """
    cam = GradCAM(model=baseline_model,
                  target_layers=[baseline_model.backbone.layer4[-1]])
    grayscale_cam = cam(input_tensor=image, targets=None)
    return grayscale_cam  # [H, W] 热力图
```

### 6.3 两种解释的对比展示
对同一张测试图像，并排展示：
- **左侧**：Grad-CAM 热力图（红=高关注区域）—— "模型在看哪里"
- **右侧**：CBM 概念解释列表 —— "模型为什么这样判断"

---

## 七、量化评估方案

### 7.1 准确率-可解释性折衷
| 模型 | 分类准确率 | 可解释性级别 |
|------|----------|------------|
| 黑盒 ResNet-18 | ~85-88% | 无（需事后解释） |
| CBM（两阶段训练） | ~80-84% | 高（概念级透明） |
| CBM（端到端微调） | ~83-86% | 中高 |

### 7.2 概念预测质量
- **整体概念准确率**：所有概念的平均二分类准确率
- **概念级 AUC**：每个概念的 ROC-AUC，取平均
- **混淆分析**：哪些概念预测困难？为什么？（可能与视觉特征不够显著有关）

### 7.3 干预实验（因果验证核心）
```python
def intervention_experiment(model, test_loader, attribute_names, device):
    """
    系统性干预实验：
    1. 基线：不做干预，记录准确率
    2. 随机干预：随机翻转 k 个概念的激活值
    3. 针对性干预：翻转预测错误的关键概念（模拟人工纠错）
    4. 完美干预：用真实属性标注替换所有概念
    """
    results = {
        'baseline': [],           # 无干预
        'random_flip_10': [],     # 随机翻转10%概念
        'random_flip_50': [],     # 随机翻转50%概念
        'top5_intervene': [],     # 纠正前5个最重要概念
        'all_intervene': [],      # 用真实概念替换（完美干预）
    }

    for images, true_concepts, true_labels in test_loader:
        images = images.to(device)

        # 1. 基线预测
        pred_concepts, pred_logits = model(images)
        baseline_acc = (pred_logits.argmax(1) == true_labels.to(device)).float().mean()
        results['baseline'].append(baseline_acc.item())

        # 2. 完美干预（用真实概念替换）
        _, intervened_logits = model(images, intervene_concepts=true_concepts.to(device))
        perfect_acc = (intervened_logits.argmax(1) == true_labels.to(device)).float().mean()
        results['all_intervene'].append(perfect_acc.item())

        # 3. 随机翻转 10%
        mask_10 = torch.rand_like(pred_concepts) < 0.1
        flipped_10 = torch.where(mask_10, 1 - pred_concepts, pred_concepts)
        _, logits_10 = model(images, intervene_concepts=flipped_10)
        results['random_flip_10'].append(
            (logits_10.argmax(1) == true_labels.to(device)).float().mean().item())

        # 4. 随机翻转 50%
        mask_50 = torch.rand_like(pred_concepts) < 0.5
        flipped_50 = torch.where(mask_50, 1 - pred_concepts, pred_concepts)
        _, logits_50 = model(images, intervene_concepts=flipped_50)
        results['random_flip_50'].append(
            (logits_50.argmax(1) == true_labels.to(device)).float().mean().item())

        # 5. 纠正前5个最重要概念
        class_weights = model.label_predictor.linear.weight[pred_logits.argmax(1)]
        importance = pred_concepts * class_weights.abs()
        _, top5_idx = importance.topk(5, dim=1)
        corrected = pred_concepts.clone()
        for i in range(images.size(0)):
            corrected[i, top5_idx[i]] = true_concepts[i, top5_idx[i]].to(device)
        _, logits_top5 = model(images, intervene_concepts=corrected)
        results['top5_intervene'].append(
            (logits_top5.argmax(1) == true_labels.to(device)).float().mean().item())

    return {k: np.mean(v) for k, v in results.items()}
```

**预期结果：**
| 干预方式 | 准确率 | 说明 |
|---------|--------|------|
| 无干预（基线） | ~82% | CBM 正常预测 |
| 随机翻转 10% | ~78% ↓ | 破坏少量概念已降低准确率 |
| 随机翻转 50% | ~60% ↓↓ | 大量概念被破坏，准确率骤降 |
| 纠正 Top-5 概念 | ~88% ↑ | 仅纠正 5 个关键概念即显著提升 |
| 完美干预 | ~95% ↑↑ | 所有概念正确时接近完美 |

**分析要点：**
- 随机翻转降低准确率 → 证明概念具有**因果性**，不仅仅是相关性
- 仅纠正 Top-5 即可大幅提升 → 证明模型学会了**稀疏依赖**关键概念
- 完美干预接近 100% → 证明概念瓶颈的信息保留是充分的

### 7.4 概念忠实度评估
```python
def concept_fidelity(model, test_loader, device):
    """
    评估概念解释是否忠实反映模型的实际推理过程。

    方法：移除 top-k 重要概念，观察预测变化。
    如果预测确实发生预期变化，说明概念忠实。
    """
    fidelity_scores = []
    for images, true_concepts, true_labels in test_loader:
        pred_concepts, pred_logits = model(images.to(device))
        pred_class = pred_logits.argmax(1)

        # 获取类别对应的概念权重
        for i in range(images.size(0)):
            class_weights = model.label_predictor.linear.weight[pred_class[i]]
            importance = pred_concepts[i] * class_weights.abs()
            _, top5_idx = importance.topk(5)

            # 将 top-5 概念置零
            masked_concepts = pred_concepts[i].clone()
            masked_concepts[top5_idx] = 0.0

            # 观察预测是否改变
            _, new_logits = model(images[i:i+1].to(device),
                                  intervene_concepts=masked_concepts.unsqueeze(0))
            new_class = new_logits.argmax(1)
            # 如果移除关键概念后预测改变了，说明这些概念确实关键
            fidelity_scores.append((new_class != pred_class[i]).float().item())

    return np.mean(fidelity_scores)
```

---

## 八、全局解释与局部解释

### 8.1 全局解释：模型整体学到了什么？
```python
def global_explanation(label_predictor, attribute_names, class_names):
    """
    分析标签预测器的权重矩阵 W [num_classes, num_concepts]。
    对每个类别，提取权重绝对值最大的 top-5 概念。

    这揭示了模型的全局决策策略：每个类别依赖哪些概念。
    """
    W = label_predictor.linear.weight.data.cpu()  # [24, num_concepts]

    global_report = {}
    for c, class_name in enumerate(class_names):
        _, top_idx = W[c].abs().topk(5)
        key_concepts = [(attribute_names[j], W[c, j].item()) for j in top_idx]
        global_report[class_name] = key_concepts

    return global_report
```

**输出示例（全局解释）：**
```
Baltimore Oriole 的关键区分概念:
  +2.31  has_wing_color_black
  +1.89  has_belly_color_orange
  +1.45  has_tail_pattern_solid
  -1.22  has_back_color_brown  (负相关：没有棕色背部)
  +0.98  has_bill_shape_cone
```

### 8.2 局部解释：单张图像为什么这样预测？
即第六节的 `generate_concept_explanation`，针对每张图给出具体的概念激活值和重要性。

---

## 九、可视化与报告图表

### 9.1 需生成的图表
1. **训练曲线**：概念预测准确率 & 分类准确率 vs Epoch
2. **混淆矩阵**：24类的分类混淆矩阵（CBM vs 基线）
3. **概念热力图**：标签预测器权重矩阵的可视化（24类 × num_concepts）
4. **干预实验曲线**：干预比例 vs 分类准确率的折线图
5. **对比展示**：同一图像的 Grad-CAM 热力图 vs CBM 概念解释
6. **准确率-可解释性 Pareto 前沿图**：展示 CBM 在准确率与可解释性之间的权衡

### 9.2 关键讨论点（报告素材）
1. **内在 vs 事后可解释性**：CBM 的解释与推理一体，不需要额外的归因步骤；Grad-CAM 只是"看哪里"，不解释"为什么"
2. **准确率-可解释性折衷**：CBM 准确率下降 2-5%，但获得了概念级的完整解释能力
3. **因果性验证**：干预实验证明概念对预测有因果影响，而非简单相关性
4. **概念忠实度**：移除关键概念后预测确实改变，说明解释忠实于模型推理
5. **稀疏性**：每个类别仅依赖 5-10 个关键概念，符合人类认知习惯

---

## 十、实现步骤（Task 分解）

### Step 1: 配置与数据加载
- 文件: `config.py`, `dataset.py`
- 内容: 定义全局超参数、类别筛选列表、CUB 数据集加载器
- 验收: 能正确加载图像 + 属性标注 + 类别标签

### Step 2: 黑盒基线模型
- 文件: `models/baseline.py`, `train_baseline.py`
- 内容: ResNet-18 迁移学习，24 类分类
- 验收: 测试集准确率 > 80%

### Step 3: 概念预测器
- 文件: `models/concept_predictor.py`, `train_concept.py`
- 内容: ResNet-18 编码器 + 概念预测头，用属性标注训练
- 验收: 平均概念预测准确率 > 85%

### Step 4: 标签预测器
- 文件: `models/label_predictor.py`, `train_label.py`
- 内容: 稀疏线性层，冻结概念预测器后训练
- 验收: 分类准确率与基线差距 < 5%

### Step 5: 完整 CBM 模型
- 文件: `models/cbm.py`
- 内容: 整合概念预测器 + 标签预测器，支持干预接口
- 验收: 前向传播正确，支持 intervene_concepts 参数

### Step 6: 解释生成
- 文件: `explain.py`
- 内容: 概念级解释 + Grad-CAM 热力图 + 对比可视化
- 验收: 能对测试图像生成自然语言解释和热力图

### Step 7: 量化评估
- 文件: `evaluate.py`, `analyze.py`
- 内容: 准确率对比、干预实验、概念忠实度、可视化图表
- 验收: 生成完整的评估报告和图表

---

## 十一、验证方案

1. **数据验证**：加载 CUB 数据集后打印类别分布、属性统计
2. **模型验证**：基线模型和 CBM 各自训练完成后，在测试集上评估准确率
3. **概念验证**：检查概念预测器对"红色翅膀"、"分叉尾巴"等直观概念的预测准确率
4. **干预验证**：运行干预实验，确认准确率变化趋势符合预期
5. **可视化验证**：生成的解释和热力图在语义上合理

---

## 十二、依赖环境

```
Python 3.10+
PyTorch >= 2.0
torchvision >= 0.15
pytorch-grad-cam >= 1.4
numpy
matplotlib
scikit-learn
Pillow
tqdm
```
