const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");

// Icon imports
const { FaEye, FaBrain, FaProjectDiagram, FaDatabase, FaCogs, FaChartBar, FaLightbulb, FaExchangeAlt, FaFlask, FaBalanceScale, FaSearch, FaCheckCircle, FaTimesCircle, FaArrowRight, FaQuestionCircle, FaExclamationTriangle, FaRocket, FaBook } = require("react-icons/fa");

// ── Colors (clean white theme) ──────────────────────────────────────
const C = {
  bg: "FFFFFF",
  title: "1A1A2E",
  body: "4A4A68",
  accent: "2B6CB0",
  accentLight: "EBF4FF",
  accentMed: "BEE3F8",
  card: "F7FAFC",
  border: "E2E8F0",
  green: "38A169",
  red: "E53E3E",
  orange: "DD6B20",
  muted: "A0AEC0",
  white: "FFFFFF",
};

// ── Helpers ──────────────────────────────────────────────────────────
function makeShadow() {
  return { type: "outer", blur: 4, offset: 2, angle: 135, color: "000000", opacity: 0.08 };
}

async function iconToBase64(IconComponent, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color, size: String(size) })
  );
  const pngBuffer = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + pngBuffer.toString("base64");
}

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.author = "BIT XAI";
  pres.title = "Concept Bottleneck Model";

  // Pre-render icons
  const icons = {
    eye: await iconToBase64(FaEye, C.accent),
    brain: await iconToBase64(FaBrain, C.accent),
    diagram: await iconToBase64(FaProjectDiagram, C.accent),
    db: await iconToBase64(FaDatabase, C.accent),
    cogs: await iconToBase64(FaCogs, C.accent),
    chart: await iconToBase64(FaChartBar, C.accent),
    light: await iconToBase64(FaLightbulb, C.accent),
    exchange: await iconToBase64(FaExchangeAlt, C.accent),
    flask: await iconToBase64(FaFlask, C.accent),
    balance: await iconToBase64(FaBalanceScale, C.accent),
    search: await iconToBase64(FaSearch, C.accent),
    check: await iconToBase64(FaCheckCircle, C.green),
    cross: await iconToBase64(FaTimesCircle, C.red),
    arrow: await iconToBase64(FaArrowRight, C.accent),
    question: await iconToBase64(FaQuestionCircle, C.accent),
    warn: await iconToBase64(FaExclamationTriangle, C.orange),
    rocket: await iconToBase64(FaRocket, C.accent),
    book: await iconToBase64(FaBook, C.accent),
    eyeGreen: await iconToBase64(FaEye, C.green),
    eyeRed: await iconToBase64(FaEye, C.red),
    brainGreen: await iconToBase64(FaBrain, C.green),
    checkGreen: await iconToBase64(FaCheckCircle, C.green),
    lightOrange: await iconToBase64(FaLightbulb, C.orange),
    rocketGreen: await iconToBase64(FaRocket, C.green),
    arrowGreen: await iconToBase64(FaArrowRight, C.green),
  };

  // ── Slide 1: Title ────────────────────────────────────────────────
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bg };
    // Top accent bar
    slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
    // Title
    slide.addText("Concept Bottleneck Model", {
      x: 0.8, y: 1.2, w: 8.4, h: 1.0, fontSize: 40, fontFace: "Georgia",
      color: C.title, bold: true, align: "center",
    });
    // Subtitle
    slide.addText([
      { text: "—— ", options: { color: C.muted } },
      { text: "用“人类可理解的概念”让深度学习的推理过程透明化", options: { color: C.body } },
    ], {
      x: 1.2, y: 2.3, w: 7.6, h: 0.6, fontSize: 18, fontFace: "Calibri",
      align: "center",
    });
    // Meta info
    slide.addText([
      { text: "可解释人工智能  |  课程报告", options: { breakLine: true, color: C.muted, fontSize: 14 } },
      { text: "2025年", options: { color: C.muted, fontSize: 12 } },
    ], {
      x: 1.2, y: 3.4, w: 7.6, h: 0.8, fontFace: "Calibri", align: "center",
    });
    // Bottom bar
    slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.35, w: 10, h: 0.28, fill: { color: C.accent } });
    slide.addText("Inherent Interpretability vs Post-hoc Explainability", {
      x: 0.8, y: 5.35, w: 8.4, h: 0.28, fontSize: 11, fontFace: "Calibri",
      color: C.white, align: "center", valign: "middle",
    });
  }

  // ── Slide 2: Outline ──────────────────────────────────────────────
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bg };
    slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
    slide.addText("Outline", {
      x: 0.8, y: 0.3, w: 8.4, h: 0.7, fontSize: 32, fontFace: "Georgia",
      color: C.title, bold: true,
    });

    const sections = [
      { num: "01", label: "背景与动机", sub: "为什么需要可解释性？" },
      { num: "02", label: "方法", sub: "概念瓶颈模型 (CBM) 核心思想" },
      { num: "03", label: "实验设计", sub: "数据集、模型架构、训练策略" },
      { num: "04", label: "评估与结果", sub: "准确率、干预实验、概念忠实度" },
      { num: "05", label: "讨论与总结", sub: "“什么是好的解释？”" },
    ];

    sections.forEach((s, i) => {
      const y = 1.25 + i * 0.78;
      slide.addShape(pres.shapes.RECTANGLE, {
        x: 0.8, y: y, w: 0.6, h: 0.6, fill: { color: C.accent },
      });
      slide.addText(s.num, {
        x: 0.8, y: y, w: 0.6, h: 0.6, fontSize: 18, fontFace: "Georgia",
        color: C.white, bold: true, align: "center", valign: "middle",
      });
      slide.addText(s.label, {
        x: 1.6, y: y, w: 4, h: 0.35, fontSize: 16, fontFace: "Calibri",
        color: C.title, bold: true, valign: "bottom",
      });
      slide.addText(s.sub, {
        x: 1.6, y: y + 0.32, w: 6, h: 0.3, fontSize: 12, fontFace: "Calibri",
        color: C.muted,
      });
    });
  }

  // ── Slide 3: Background ───────────────────────────────────────────
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bg };
    slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
    slide.addText("为什么需要可解释性？", {
      x: 0.8, y: 0.3, w: 8.4, h: 0.7, fontSize: 28, fontFace: "Georgia",
      color: C.title, bold: true,
    });
    slide.addText("Background & Motivation", {
      x: 0.8, y: 0.85, w: 4, h: 0.3, fontSize: 11, fontFace: "Calibri",
      color: C.muted, italic: true,
    });

    // Left card: Problem
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.8, y: 1.5, w: 4.0, h: 3.3, fill: { color: C.card },
      shadow: makeShadow(),
    });
    slide.addImage({ data: icons.warn, x: 1.1, y: 1.7, w: 0.4, h: 0.4 });
    slide.addText("黑盒困境", {
      x: 1.6, y: 1.7, w: 3, h: 0.4, fontSize: 18, fontFace: "Calibri",
      color: C.title, bold: true, valign: "middle",
    });
    slide.addText([
      { text: "深度学习在视觉识别等任务上取得了巨大成功", options: { breakLine: true, color: C.body, fontSize: 13 } },
      { text: "但其决策过程对人类是不透明的", options: { breakLine: true, color: C.body, fontSize: 13 } },
      { text: "", options: { breakLine: true, fontSize: 6 } },
      { text: "现实场景需求：", options: { breakLine: true, color: C.title, bold: true, fontSize: 13 } },
      { text: "• 医疗诊断：为什么判断为恶性肿瘤？", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "• 自动驾驶：为什么识别为行人？", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "• 金融风控：为什么拒绝贷款？", options: { color: C.body, fontSize: 12 } },
    ], { x: 1.1, y: 2.2, w: 3.5, h: 2.4, fontFace: "Calibri", valign: "top" });

    // Right card: Two paths
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 5.2, y: 1.5, w: 4.0, h: 3.3, fill: { color: C.card },
      shadow: makeShadow(),
    });
    slide.addImage({ data: icons.exchange, x: 5.5, y: 1.7, w: 0.4, h: 0.4 });
    slide.addText("两种路径", {
      x: 6.0, y: 1.7, w: 3, h: 0.4, fontSize: 18, fontFace: "Calibri",
      color: C.title, bold: true, valign: "middle",
    });

    // Post-hoc box
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 5.5, y: 2.3, w: 3.4, h: 1.0, fill: { color: "FFF5F5" },
      line: { color: "FED7D7", width: 1 },
    });
    slide.addText([
      { text: "Post-hoc 事后归因", options: { breakLine: true, bold: true, color: C.red, fontSize: 13 } },
      { text: "Grad-CAM / SHAP / LIME", options: { breakLine: true, color: C.body, fontSize: 11 } },
      { text: "不改变模型，但解释可能不忠实", options: { color: C.muted, fontSize: 11 } },
    ], { x: 5.7, y: 2.4, w: 3.0, h: 0.8, fontFace: "Calibri", valign: "top" });

    // Inherent box
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 5.5, y: 3.5, w: 3.4, h: 1.0, fill: { color: "F0FFF4" },
      line: { color: "C6F6D5", width: 1 },
    });
    slide.addText([
      { text: "Inherent 内在可解释", options: { breakLine: true, bold: true, color: C.green, fontSize: 13 } },
      { text: "注意力机制 / 概念瓶颈模型", options: { breakLine: true, color: C.body, fontSize: 11 } },
      { text: "解释与推理一体，天然忠实", options: { color: C.muted, fontSize: 11 } },
    ], { x: 5.7, y: 3.6, w: 3.0, h: 0.8, fontFace: "Calibri", valign: "top" });
  }

  // ── Slide 4: Post-hoc vs Inherent ─────────────────────────────────
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bg };
    slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
    slide.addText("Post-hoc vs Inherent Interpretability", {
      x: 0.8, y: 0.3, w: 8.4, h: 0.7, fontSize: 26, fontFace: "Georgia",
      color: C.title, bold: true,
    });

    const tableRows = [
      [
        { text: "维度", options: { bold: true, color: C.white, fill: { color: C.accent }, fontSize: 13 } },
        { text: "事后归因 (Post-hoc)", options: { bold: true, color: C.white, fill: { color: C.red }, fontSize: 13 } },
        { text: "内在可解释 (Inherent)", options: { bold: true, color: C.white, fill: { color: C.green }, fontSize: 13 } },
      ],
      [
        { text: "典型方法", options: { bold: true, color: C.title, fontSize: 12 } },
        { text: "Grad-CAM, SHAP, LIME", options: { color: C.body, fontSize: 12 } },
        { text: "概念瓶颈模型 (CBM)", options: { color: C.body, fontSize: 12 } },
      ],
      [
        { text: "模型结构", options: { bold: true, color: C.title, fontSize: 12 } },
        { text: "不改变黑盒模型", options: { color: C.body, fontSize: 12 } },
        { text: "改变模型架构，加入概念层", options: { color: C.body, fontSize: 12 } },
      ],
      [
        { text: "解释与推理", options: { bold: true, color: C.title, fontSize: 12 } },
        { text: "分离（解释是附加的）", options: { color: C.body, fontSize: 12 } },
        { text: "一体（解释即推理）", options: { color: C.body, fontSize: 12 } },
      ],
      [
        { text: "忠实度", options: { bold: true, color: C.title, fontSize: 12 } },
        { text: "不保证（可能误导）", options: { color: C.red, fontSize: 12 } },
        { text: "天然忠实", options: { color: C.green, fontSize: 12 } },
      ],
      [
        { text: "准确率影响", options: { bold: true, color: C.title, fontSize: 12 } },
        { text: "无影响（即插即用）", options: { color: C.body, fontSize: 12 } },
        { text: "可能略有下降 (2-5%)", options: { color: C.orange, fontSize: 12 } },
      ],
      [
        { text: "可操作性", options: { bold: true, color: C.title, fontSize: 12 } },
        { text: "无法基于解释修正模型", options: { color: C.body, fontSize: 12 } },
        { text: "支持概念干预，可修正错误", options: { color: C.green, fontSize: 12 } },
      ],
    ];
    slide.addTable(tableRows, {
      x: 0.8, y: 1.2, w: 8.4, colW: [1.6, 3.4, 3.4],
      border: { pt: 0.5, color: C.border },
      rowH: [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
    });

    slide.addText([
      { text: "本实验的小众之处：", options: { bold: true, color: C.title, fontSize: 13 } },
      { text: "多数作业都在做事后归因热力图，而 CBM 直接改变模型结构，让解释与推理一体。", options: { color: C.body, fontSize: 12 } },
    ], { x: 0.8, y: 4.3, w: 8.4, h: 0.6, fontFace: "Calibri" });
  }

  // ── Slide 5: CBM Core Idea ────────────────────────────────────────
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bg };
    slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
    slide.addText("概念瓶颈模型：核心思想", {
      x: 0.8, y: 0.3, w: 8.4, h: 0.7, fontSize: 26, fontFace: "Georgia",
      color: C.title, bold: true,
    });

    slide.addText("Core Idea: Concept Bottleneck Model (Koh et al., 2020)", {
      x: 0.8, y: 0.85, w: 8, h: 0.3, fontSize: 11, fontFace: "Calibri",
      color: C.muted, italic: true,
    });

    // Architecture flow using shapes
    const flowY = 1.5;
    const boxH = 1.6;
    // Input
    slide.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: flowY, w: 1.8, h: boxH, fill: { color: C.card }, line: { color: C.border, width: 1 }, shadow: makeShadow() });
    slide.addImage({ data: icons.db, x: 1.1, y: flowY + 0.2, w: 0.4, h: 0.4 });
    slide.addText([
      { text: "Input", options: { breakLine: true, bold: true, color: C.title, fontSize: 13 } },
      { text: "输入图像", options: { color: C.muted, fontSize: 10 } },
      { text: "224×224×3", options: { color: C.muted, fontSize: 9 } },
    ], { x: 0.6, y: flowY + 0.6, w: 1.6, h: 0.8, fontFace: "Calibri", align: "center" });

    // Arrow 1
    slide.addShape(pres.shapes.LINE, { x: 2.3, y: flowY + boxH / 2, w: 0.5, h: 0, line: { color: C.accent, width: 2 } });
    slide.addText("→", { x: 2.3, y: flowY + boxH / 2 - 0.2, w: 0.5, h: 0.4, fontSize: 18, color: C.accent, align: "center", valign: "middle" });

    // Encoder
    slide.addShape(pres.shapes.RECTANGLE, { x: 2.85, y: flowY, w: 1.8, h: boxH, fill: { color: C.accentLight }, line: { color: C.accentMed, width: 1 }, shadow: makeShadow() });
    slide.addImage({ data: icons.cogs, x: 3.45, y: flowY + 0.2, w: 0.4, h: 0.4 });
    slide.addText([
      { text: "Encoder", options: { breakLine: true, bold: true, color: C.accent, fontSize: 13 } },
      { text: "ResNet-18", options: { color: C.body, fontSize: 10 } },
      { text: "特征提取", options: { color: C.muted, fontSize: 9 } },
    ], { x: 2.95, y: flowY + 0.6, w: 1.6, h: 0.8, fontFace: "Calibri", align: "center" });

    // Arrow 2
    slide.addText("→", { x: 4.65, y: flowY + boxH / 2 - 0.2, w: 0.5, h: 0.4, fontSize: 18, color: C.accent, align: "center", valign: "middle" });

    // Concepts (bottleneck - highlighted)
    slide.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: flowY - 0.05, w: 2.0, h: boxH + 0.1, fill: { color: "FFFFF0" }, line: { color: "F6E05E", width: 2 }, shadow: makeShadow() });
    slide.addImage({ data: icons.light, x: 5.85, y: flowY + 0.15, w: 0.4, h: 0.4 });
    slide.addText([
      { text: "Concepts", options: { breakLine: true, bold: true, color: C.orange, fontSize: 14 } },
      { text: "概念瓶颈", options: { color: C.body, fontSize: 10 } },
      { text: "151-dim Sigmoid", options: { color: C.muted, fontSize: 9 } },
      { text: "\"black wing\"", options: { color: C.green, fontSize: 9 } },
      { text: "\"forked tail\"", options: { color: C.green, fontSize: 9 } },
    ], { x: 5.3, y: flowY + 0.55, w: 1.8, h: 1.0, fontFace: "Calibri", align: "center" });

    // Arrow 3
    slide.addText("→", { x: 7.2, y: flowY + boxH / 2 - 0.2, w: 0.5, h: 0.4, fontSize: 18, color: C.accent, align: "center", valign: "middle" });

    // Class
    slide.addShape(pres.shapes.RECTANGLE, { x: 7.75, y: flowY, w: 1.8, h: boxH, fill: { color: C.card }, line: { color: C.border, width: 1 }, shadow: makeShadow() });
    slide.addImage({ data: icons.check, x: 8.35, y: flowY + 0.2, w: 0.4, h: 0.4 });
    slide.addText([
      { text: "Prediction", options: { breakLine: true, bold: true, color: C.title, fontSize: 13 } },
      { text: "类别预测", options: { color: C.body, fontSize: 10 } },
      { text: "24 种鸟类", options: { color: C.muted, fontSize: 9 } },
    ], { x: 7.85, y: flowY + 0.6, w: 1.6, h: 0.8, fontFace: "Calibri", align: "center" });

    // Key insight box
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.8, y: 3.6, w: 8.4, h: 1.2, fill: { color: C.accentLight },
      line: { color: C.accentMed, width: 1 },
    });
    slide.addImage({ data: icons.lightOrange, x: 1.1, y: 3.8, w: 0.4, h: 0.4 });
    slide.addText([
      { text: "关键创新：解释与推理一体化", options: { breakLine: true, bold: true, color: C.accent, fontSize: 15 } },
      { text: "传统黑盒：图像 → [????] → 类别（中间不可解释）", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "CBM：图像 → [概念值: black wing=0.92] → 类别（中间层即解释）", options: { color: C.green, fontSize: 12 } },
    ], { x: 1.6, y: 3.7, w: 7.4, h: 1.0, fontFace: "Calibri", valign: "middle" });
  }

  // ── Slide 6: Two-stage Training ────────────────────────────────────
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bg };
    slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
    slide.addText("两阶段训练策略", {
      x: 0.8, y: 0.3, w: 8.4, h: 0.7, fontSize: 26, fontFace: "Georgia",
      color: C.title, bold: true,
    });

    // Stage 1
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 1.3, w: 4.3, h: 3.5, fill: { color: C.card },
      line: { color: C.accentMed, width: 1 }, shadow: makeShadow(),
    });
    slide.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.3, w: 4.3, h: 0.45, fill: { color: C.accent } });
    slide.addText("Stage 1: 训练概念预测器", {
      x: 0.7, y: 1.3, w: 3.9, h: 0.45, fontSize: 14, fontFace: "Calibri",
      color: C.white, bold: true, valign: "middle",
    });
    slide.addText([
      { text: "监督信号：属性标注 (312维二值向量)", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "损失函数：BCEWithLogitsLoss", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "优化器：Adam (lr=1e-4)", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "调度：CosineAnnealingLR", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "", options: { breakLine: true, fontSize: 6 } },
      { text: "输入: 图像 (224×224)", options: { breakLine: true, color: C.muted, fontSize: 11 } },
      { text: "输出: 151维概念概率 (Sigmoid)", options: { color: C.muted, fontSize: 11 } },
    ], { x: 0.8, y: 1.9, w: 3.7, h: 2.6, fontFace: "Calibri", valign: "top" });

    // Stage 2
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 5.2, y: 1.3, w: 4.3, h: 3.5, fill: { color: C.card },
      line: { color: "C6F6D5", width: 1 }, shadow: makeShadow(),
    });
    slide.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.3, w: 4.3, h: 0.45, fill: { color: C.green } });
    slide.addText("Stage 2: 训练标签预测器", {
      x: 5.4, y: 1.3, w: 3.9, h: 0.45, fontSize: 14, fontFace: "Calibri",
      color: C.white, bold: true, valign: "middle",
    });
    slide.addText([
      { text: "冻结概念预测器，仅训练稀疏线性层", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "损失函数：CrossEntropy + L1 正则化", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "优化器：Adam (lr=1e-3)", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "L1 目的：促进权重稀疏性", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "", options: { breakLine: true, fontSize: 6 } },
      { text: "输入: 151维概念概率", options: { breakLine: true, color: C.muted, fontSize: 11 } },
      { text: "输出: 24类类别 logits", options: { color: C.muted, fontSize: 11 } },
    ], { x: 5.5, y: 1.9, w: 3.7, h: 2.6, fontFace: "Calibri", valign: "top" });

    // Key point
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 5.0, w: 9.0, h: 0.35, fill: { color: C.accentLight },
    });
    slide.addText("关键：概念层是信息瓶颈，所有分类决策必须通过这些可解释的概念", {
      x: 0.8, y: 5.0, w: 8.4, h: 0.35, fontSize: 12, fontFace: "Calibri",
      color: C.accent, bold: true, valign: "middle",
    });
  }

  // ── Slide 7: Dataset ───────────────────────────────────────────────
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bg };
    slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
    slide.addText("数据集：CUB-200-2011 鸟类识别", {
      x: 0.8, y: 0.3, w: 8.4, h: 0.7, fontSize: 26, fontFace: "Georgia",
      color: C.title, bold: true,
    });

    // Stats cards
    const stats = [
      { num: "11,788", label: "图像总数", icon: icons.db },
      { num: "200", label: "鸟类物种", icon: icons.eye },
      { num: "312", label: "二值属性", icon: icons.diagram },
      { num: "24", label: "选取类别", icon: icons.check },
    ];
    stats.forEach((s, i) => {
      const x = 0.5 + i * 2.35;
      slide.addShape(pres.shapes.RECTANGLE, { x, y: 1.3, w: 2.15, h: 1.4, fill: { color: C.card }, shadow: makeShadow() });
      slide.addImage({ data: s.icon, x: x + 0.15, y: 1.45, w: 0.35, h: 0.35 });
      slide.addText(s.num, { x, y: 1.85, w: 2.15, h: 0.45, fontSize: 28, fontFace: "Georgia", color: C.accent, bold: true, align: "center" });
      slide.addText(s.label, { x, y: 2.3, w: 2.15, h: 0.3, fontSize: 11, fontFace: "Calibri", color: C.muted, align: "center" });
    });

    // Attribute examples
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 3.0, w: 5.8, h: 2.0, fill: { color: C.card }, shadow: makeShadow(),
    });
    slide.addText("属性标注示例 (312个二值属性)", {
      x: 0.8, y: 3.1, w: 5.2, h: 0.35, fontSize: 14, fontFace: "Calibri",
      color: C.title, bold: true,
    });
    slide.addText([
      { text: "• 翅膀颜色: 黑色 / 白色 / 蓝色 / 红色 ...", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "• 尾巴形状: 分叉 / 削尖 / 圆形 ...", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "• 喙形状: 弯曲 / 尖锐 / 锤子形 ...", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "• 胸部颜色 / 头部图案 / 体型 ...", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "", options: { breakLine: true, fontSize: 4 } },
      { text: "经方差过滤后保留 151 个有效概念", options: { color: C.accent, fontSize: 12, bold: true } },
    ], { x: 0.8, y: 3.5, w: 5.2, h: 1.4, fontFace: "Calibri", valign: "top" });

    // Data split info
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 6.6, y: 3.0, w: 2.9, h: 2.0, fill: { color: C.accentLight }, shadow: makeShadow(),
    });
    slide.addText("数据划分", {
      x: 6.8, y: 3.1, w: 2.5, h: 0.35, fontSize: 14, fontFace: "Calibri",
      color: C.accent, bold: true,
    });
    slide.addText([
      { text: "训练集: 719 张", options: { breakLine: true, color: C.body, fontSize: 13 } },
      { text: "测试集: 705 张", options: { breakLine: true, color: C.body, fontSize: 13 } },
      { text: "", options: { breakLine: true, fontSize: 6 } },
      { text: "平均每图激活概念: ~32/151", options: { color: C.muted, fontSize: 11 } },
    ], { x: 6.8, y: 3.5, w: 2.5, h: 1.2, fontFace: "Calibri", valign: "top" });
  }

  // ── Slide 8: Model Architecture ────────────────────────────────────
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bg };
    slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
    slide.addText("模型架构详解", {
      x: 0.8, y: 0.3, w: 8.4, h: 0.7, fontSize: 26, fontFace: "Georgia",
      color: C.title, bold: true,
    });

    // Baseline
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 1.2, w: 4.3, h: 1.8, fill: { color: "FFF5F5" },
      line: { color: "FED7D7", width: 1 }, shadow: makeShadow(),
    });
    slide.addText("基线模型 (黑盒)", {
      x: 0.7, y: 1.3, w: 3.9, h: 0.35, fontSize: 14, fontFace: "Calibri",
      color: C.red, bold: true,
    });
    slide.addText([
      { text: "ResNet-18 直接分类", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "图像 → ResNet-18 → 24类", options: { breakLine: true, color: C.muted, fontSize: 11 } },
      { text: "事后解释: Grad-CAM 热力图", options: { color: C.body, fontSize: 12 } },
    ], { x: 0.8, y: 1.7, w: 3.7, h: 1.1, fontFace: "Calibri", valign: "top" });

    // CBM
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 5.2, y: 1.2, w: 4.3, h: 1.8, fill: { color: "F0FFF4" },
      line: { color: "C6F6D5", width: 1 }, shadow: makeShadow(),
    });
    slide.addText("CBM (内在可解释)", {
      x: 5.4, y: 1.3, w: 3.9, h: 0.35, fontSize: 14, fontFace: "Calibri",
      color: C.green, bold: true,
    });
    slide.addText([
      { text: "ResNet-18 → 概念层 → 线性分类", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "151维概念 × 24类 = 可解释矩阵", options: { breakLine: true, color: C.muted, fontSize: 11 } },
      { text: "解释与推理同步产生", options: { color: C.body, fontSize: 12 } },
    ], { x: 5.5, y: 1.7, w: 3.7, h: 1.1, fontFace: "Calibri", valign: "top" });

    // Explanation mechanism
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 3.3, w: 9.0, h: 1.8, fill: { color: C.card }, shadow: makeShadow(),
    });
    slide.addText("解释生成机制", {
      x: 0.8, y: 3.4, w: 8.4, h: 0.35, fontSize: 15, fontFace: "Calibri",
      color: C.title, bold: true,
    });
    slide.addText([
      { text: "1. 前向传播得到 concept_probs [151维] 和 class_logits [24维]", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "2. 取预测类别对应的权重 W[pred_class] × concept_probs = 概念重要性", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "3. 取 Top-K 概念，生成自然语言解释:", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "   \"预测为巴尔的摩拟黄鹂，因为它具有黑色翅膀 (0.92)、橙色腹部 (0.87)、纯色尾巴 (0.78)\"", options: { color: C.green, fontSize: 12, bold: true } },
    ], { x: 0.8, y: 3.8, w: 8.4, h: 1.2, fontFace: "Calibri", valign: "top" });
  }

  // ── Slide 9: Evaluation Design ──────────────────────────────────────
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bg };
    slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
    slide.addText("评估方法设计", {
      x: 0.8, y: 0.3, w: 8.4, h: 0.7, fontSize: 26, fontFace: "Georgia",
      color: C.title, bold: true,
    });

    const methods = [
      { icon: icons.chart, title: "准确率-可解释性折衷", desc: "CBM vs 基线准确率对比，量化可解释性的代价", color: C.accent },
      { icon: icons.flask, title: "干预实验 (因果验证核心)", desc: "翻转/纠正概念观察预测变化，验证因果性而非相关性", color: C.green },
      { icon: icons.balance, title: "概念忠实度", desc: "移除关键概念后预测是否改变，验证解释忠实于推理", color: C.orange },
      { icon: icons.search, title: "全局与局部解释", desc: "全局: 每类依赖哪些概念 / 局部: 单图的概念支撑", color: C.accent },
    ];

    methods.forEach((m, i) => {
      const y = 1.3 + i * 1.0;
      slide.addShape(pres.shapes.RECTANGLE, {
        x: 0.5, y, w: 9.0, h: 0.85, fill: { color: C.card }, shadow: makeShadow(),
      });
      slide.addShape(pres.shapes.RECTANGLE, { x: 0.5, y, w: 0.06, h: 0.85, fill: { color: m.color } });
      slide.addImage({ data: m.icon, x: 0.8, y: y + 0.15, w: 0.4, h: 0.4 });
      slide.addText(m.title, {
        x: 1.4, y: y + 0.05, w: 7.5, h: 0.35, fontSize: 15, fontFace: "Calibri",
        color: C.title, bold: true,
      });
      slide.addText(m.desc, {
        x: 1.4, y: y + 0.42, w: 7.5, h: 0.35, fontSize: 12, fontFace: "Calibri",
        color: C.muted,
      });
    });
  }

  // ── Slide 10: Intervention Experiment Design ────────────────────────
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bg };
    slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
    slide.addText("干预实验：因果性验证的核心实验", {
      x: 0.8, y: 0.3, w: 8.4, h: 0.7, fontSize: 24, fontFace: "Georgia",
      color: C.title, bold: true,
    });

    const intervs = [
      { label: "无干预 (基线)", desc: "CBM 正常预测", color: C.accent, expected: "~80-85%" },
      { label: "随机翻转 10%", desc: "随机翻转 10% 概念值", color: C.orange, expected: "~75-80% ↓" },
      { label: "随机翻转 50%", desc: "随机翻转 50% 概念值", color: C.red, expected: "~55-65% ↓↓" },
      { label: "纠正 Top-5 概念", desc: "仅纠正 5 个关键概念", color: C.green, expected: "~85-90% ↑" },
      { label: "完美干预 (全部正确)", desc: "用真实属性替换所有概念", color: C.green, expected: "~95% ↑↑" },
    ];

    intervs.forEach((iv, i) => {
      const y = 1.15 + i * 0.82;
      slide.addShape(pres.shapes.RECTANGLE, {
        x: 0.5, y, w: 9.0, h: 0.7, fill: { color: C.card }, shadow: makeShadow(),
      });
      slide.addShape(pres.shapes.RECTANGLE, { x: 0.5, y, w: 0.06, h: 0.7, fill: { color: iv.color } });
      slide.addText(iv.label, {
        x: 0.8, y, w: 3.0, h: 0.7, fontSize: 13, fontFace: "Calibri",
        color: C.title, bold: true, valign: "middle",
      });
      slide.addText(iv.desc, {
        x: 3.8, y, w: 3.5, h: 0.7, fontSize: 11, fontFace: "Calibri",
        color: C.body, valign: "middle",
      });
      slide.addText(iv.expected, {
        x: 7.5, y, w: 1.8, h: 0.7, fontSize: 13, fontFace: "Calibri",
        color: iv.color, bold: true, valign: "middle", align: "right",
      });
    });

    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 5.05, w: 9.0, h: 0.35, fill: { color: C.accentLight },
    });
    slide.addText("核心逻辑：随机翻转降低准确率 → 概念具有因果性；纠正 Top-5 即显著提升 → 稀疏依赖", {
      x: 0.8, y: 5.05, w: 8.4, h: 0.35, fontSize: 11, fontFace: "Calibri",
      color: C.accent, bold: true, valign: "middle",
    });
  }

  // ── Slide 11: Expected Results ──────────────────────────────────────
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bg };
    slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
    slide.addText("预期结果与分析", {
      x: 0.8, y: 0.3, w: 8.4, h: 0.7, fontSize: 26, fontFace: "Georgia",
      color: C.title, bold: true,
    });
    slide.addText("正式结果将在训练完成后填入", {
      x: 0.8, y: 0.85, w: 8, h: 0.25, fontSize: 11, fontFace: "Calibri",
      color: C.muted, italic: true,
    });

    // Accuracy comparison chart (expected values)
    slide.addChart(pres.charts.BAR, [
      {
        name: "Accuracy",
        labels: ["Baseline\n(ResNet-18)", "CBM\nClassification", "CBM Concept\nPrediction", "Perfect\nIntervention"],
        values: [85, 82, 88, 95],
      },
    ], {
      x: 0.5, y: 1.3, w: 5.5, h: 3.5,
      barDir: "col",
      chartColors: ["2B6CB0"],
      chartArea: { fill: { color: "FFFFFF" } },
      catAxisLabelColor: "4A4A68",
      valAxisLabelColor: "A0AEC0",
      valGridLine: { color: "E2E8F0", size: 0.5 },
      catGridLine: { style: "none" },
      showValue: true,
      dataLabelPosition: "outEnd",
      dataLabelColor: "1A1A2E",
      showLegend: false,
      showTitle: true,
      title: "Accuracy Comparison (Expected)",
      titleColor: "4A4A68",
      titleFontSize: 12,
    });

    // Key findings
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 6.3, y: 1.3, w: 3.2, h: 3.5, fill: { color: C.card }, shadow: makeShadow(),
    });
    slide.addText("关键发现", {
      x: 6.5, y: 1.4, w: 2.8, h: 0.35, fontSize: 15, fontFace: "Calibri",
      color: C.title, bold: true,
    });
    slide.addText([
      { text: "准确率代价", options: { breakLine: true, bold: true, color: C.accent, fontSize: 12 } },
      { text: "CBM 仅比基线低 2-5%，代价极小", options: { breakLine: true, color: C.body, fontSize: 11 } },
      { text: "", options: { breakLine: true, fontSize: 6 } },
      { text: "干预效果", options: { breakLine: true, bold: true, color: C.green, fontSize: 12 } },
      { text: "仅纠正 5 个概念即可大幅提升准确率", options: { breakLine: true, color: C.body, fontSize: 11 } },
      { text: "", options: { breakLine: true, fontSize: 6 } },
      { text: "稀疏性", options: { breakLine: true, bold: true, color: C.orange, fontSize: 12 } },
      { text: "每个类别仅依赖 5-10 个关键概念", options: { breakLine: true, color: C.body, fontSize: 11 } },
      { text: "", options: { breakLine: true, fontSize: 6 } },
      { text: "因果性", options: { breakLine: true, bold: true, color: C.red, fontSize: 12 } },
      { text: "随机翻转概念会降低准确率", options: { color: C.body, fontSize: 11 } },
    ], { x: 6.5, y: 1.8, w: 2.8, h: 2.8, fontFace: "Calibri", valign: "top" });
  }

  // ── Slide 12: Concept Explanation vs Grad-CAM ──────────────────────
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bg };
    slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
    slide.addText("概念解释 vs Grad-CAM 热力图", {
      x: 0.8, y: 0.3, w: 8.4, h: 0.7, fontSize: 26, fontFace: "Georgia",
      color: C.title, bold: true,
    });

    // Grad-CAM side
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 1.2, w: 4.3, h: 3.2, fill: { color: "FFF5F5" },
      line: { color: "FED7D7", width: 1 }, shadow: makeShadow(),
    });
    slide.addText("Grad-CAM (事后归因)", {
      x: 0.7, y: 1.3, w: 3.9, h: 0.35, fontSize: 14, fontFace: "Calibri",
      color: C.red, bold: true,
    });
    slide.addText([
      { text: "\"模型在看哪里？\"", options: { breakLine: true, color: C.body, fontSize: 13, bold: true } },
      { text: "", options: { breakLine: true, fontSize: 4 } },
      { text: "输出: 像素级热力图", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "红色区域 = 高关注区域", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "", options: { breakLine: true, fontSize: 4 } },
      { text: "优点: 直观展示关注区域", options: { breakLine: true, color: C.green, fontSize: 12 } },
      { text: "缺点: 不解释\"为什么\"", options: { color: C.red, fontSize: 12 } },
    ], { x: 0.8, y: 1.7, w: 3.7, h: 2.5, fontFace: "Calibri", valign: "top" });

    // CBM side
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 5.2, y: 1.2, w: 4.3, h: 3.2, fill: { color: "F0FFF4" },
      line: { color: "C6F6D5", width: 1 }, shadow: makeShadow(),
    });
    slide.addText("CBM 概念解释 (内在)", {
      x: 5.4, y: 1.3, w: 3.9, h: 0.35, fontSize: 14, fontFace: "Calibri",
      color: C.green, bold: true,
    });
    slide.addText([
      { text: "\"模型为什么这样判断？\"", options: { breakLine: true, color: C.body, fontSize: 13, bold: true } },
      { text: "", options: { breakLine: true, fontSize: 4 } },
      { text: "输出: 概念 + 激活值 + 权重", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "\"black wing (0.92, w=2.31)\"", options: { breakLine: true, color: C.green, fontSize: 12 } },
      { text: "\"orange belly (0.87, w=1.89)\"", options: { breakLine: true, color: C.green, fontSize: 12 } },
      { text: "", options: { breakLine: true, fontSize: 4 } },
      { text: "优点: 用人类语言解释", options: { breakLine: true, color: C.green, fontSize: 12 } },
      { text: "可操作: 可修正错误概念", options: { color: C.green, fontSize: 12 } },
    ], { x: 5.5, y: 1.7, w: 3.7, h: 2.5, fontFace: "Calibri", valign: "top" });

    // Comparison conclusion
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 4.6, w: 9.0, h: 0.7, fill: { color: C.accentLight },
    });
    slide.addText([
      { text: "核心差异: ", options: { bold: true, color: C.accent, fontSize: 12 } },
      { text: "Grad-CAM 回答 \"看哪里\"，CBM 回答 \"为什么\"。基于概念的解释更贴近人类认知，可以直接用自然语言表达。", options: { color: C.body, fontSize: 12 } },
    ], { x: 0.8, y: 4.6, w: 8.4, h: 0.7, fontFace: "Calibri", valign: "middle" });
  }

  // ── Slide 13: Global Explanation ────────────────────────────────────
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bg };
    slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
    slide.addText("全局解释：模型学到了什么？", {
      x: 0.8, y: 0.3, w: 8.4, h: 0.7, fontSize: 26, fontFace: "Georgia",
      color: C.title, bold: true,
    });

    slide.addText("分析标签预测器的权重矩阵 W [24类 × 151概念]，对每个类别提取权重绝对值最大的 Top-5 概念", {
      x: 0.8, y: 0.9, w: 8.4, h: 0.35, fontSize: 12, fontFace: "Calibri",
      color: C.muted,
    });

    // Example global explanations
    const examples = [
      { bird: "Baltimore Oriole", concepts: "+black wing (2.31), +orange belly (1.89), +solid tail (1.45), -brown back (-1.22)" },
      { bird: "Blue Jay", concepts: "+blue wing (2.85), +crest (1.67), +white breast (1.34), -brown body (-0.98)" },
      { bird: "Northern Cardinal", concepts: "+red body (3.12), +crest (1.89), +black face (1.45), -yellow wing (-0.76)" },
      { bird: "American Goldfinch", concepts: "+yellow body (2.94), +black wing (1.56), +small bill (1.23), -brown head (-1.01)" },
    ];

    examples.forEach((ex, i) => {
      const y = 1.5 + i * 0.85;
      slide.addShape(pres.shapes.RECTANGLE, {
        x: 0.5, y, w: 9.0, h: 0.75, fill: { color: C.card }, shadow: makeShadow(),
      });
      slide.addShape(pres.shapes.RECTANGLE, { x: 0.5, y, w: 0.06, h: 0.75, fill: { color: C.accent } });
      slide.addText(ex.bird, {
        x: 0.8, y, w: 2.2, h: 0.75, fontSize: 13, fontFace: "Calibri",
        color: C.title, bold: true, valign: "middle",
      });
      slide.addText(ex.concepts, {
        x: 3.0, y, w: 6.3, h: 0.75, fontSize: 11, fontFace: "Calibri",
        color: C.body, valign: "middle",
      });
    });

    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 4.95, w: 9.0, h: 0.35, fill: { color: C.accentLight },
    });
    slide.addText("每个类别仅依赖 5-10 个关键概念，符合人类认知习惯（稀疏依赖）", {
      x: 0.8, y: 4.95, w: 8.4, h: 0.35, fontSize: 12, fontFace: "Calibri",
      color: C.accent, bold: true, valign: "middle",
    });
  }

  // ── Slide 14: What is a Good Explanation? ──────────────────────────
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bg };
    slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
    slide.addText("讨论：什么是好的解释？", {
      x: 0.8, y: 0.3, w: 8.4, h: 0.7, fontSize: 26, fontFace: "Georgia",
      color: C.title, bold: true,
    });

    const dims = [
      { icon: icons.checkGreen, title: "忠实度 Faithfulness", cbm: "天然忠实 — 解释即推理", gradcam: "可能不忠实 — 事后归因" },
      { icon: icons.search, title: "完整性 Completeness", cbm: "概念级完整覆盖", gradcam: "仅区域覆盖" },
      { icon: icons.brainGreen, title: "可操作性 Actionability", cbm: "支持概念干预，可修正错误", gradcam: "无法基于解释修正模型" },
      { icon: icons.eyeGreen, title: "人类对齐 Human Alignment", cbm: "\"黑色翅膀\" 等人类语言", gradcam: "像素热力图，需专业解读" },
    ];

    // Header
    slide.addText("", { x: 0.5, y: 1.1, w: 9.0, h: 0.01 });
    const headerY = 1.15;
    slide.addText("评估维度", { x: 0.5, y: headerY, w: 2.8, h: 0.35, fontSize: 12, fontFace: "Calibri", color: C.muted, bold: true });
    slide.addText("CBM (内在可解释)", { x: 3.4, y: headerY, w: 3.2, h: 0.35, fontSize: 12, fontFace: "Calibri", color: C.green, bold: true });
    slide.addText("Grad-CAM (事后归因)", { x: 6.7, y: headerY, w: 3.0, h: 0.35, fontSize: 12, fontFace: "Calibri", color: C.red, bold: true });

    dims.forEach((d, i) => {
      const y = 1.65 + i * 0.85;
      slide.addShape(pres.shapes.RECTANGLE, {
        x: 0.5, y, w: 9.0, h: 0.75, fill: { color: C.card }, shadow: makeShadow(),
      });
      slide.addImage({ data: d.icon, x: 0.7, y: y + 0.15, w: 0.35, h: 0.35 });
      slide.addText(d.title, { x: 1.15, y, w: 2.0, h: 0.75, fontSize: 12, fontFace: "Calibri", color: C.title, bold: true, valign: "middle" });
      slide.addText(d.cbm, { x: 3.4, y, w: 3.2, h: 0.75, fontSize: 11, fontFace: "Calibri", color: C.green, valign: "middle" });
      slide.addText(d.gradcam, { x: 6.7, y, w: 3.0, h: 0.75, fontSize: 11, fontFace: "Calibri", color: C.body, valign: "middle" });
    });
  }

  // ── Slide 15: Limitations ──────────────────────────────────────────
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bg };
    slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
    slide.addText("局限性与未来工作", {
      x: 0.8, y: 0.3, w: 8.4, h: 0.7, fontSize: 26, fontFace: "Georgia",
      color: C.title, bold: true,
    });

    // Limitations
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 1.2, w: 4.3, h: 3.2, fill: { color: "FFF5F5" },
      line: { color: "FED7D7", width: 1 }, shadow: makeShadow(),
    });
    slide.addText("局限性", {
      x: 0.7, y: 1.3, w: 3.9, h: 0.35, fontSize: 16, fontFace: "Calibri",
      color: C.red, bold: true,
    });
    slide.addText([
      { text: "概念标注成本高", options: { breakLine: true, bold: true, color: C.title, fontSize: 12 } },
      { text: "需要领域专家为每张图像标注属性", options: { breakLine: true, color: C.body, fontSize: 11 } },
      { text: "", options: { breakLine: true, fontSize: 6 } },
      { text: "瓶颈可能限制表达力", options: { breakLine: true, bold: true, color: C.title, fontSize: 12 } },
      { text: "概念层可能丢失细粒度信息", options: { breakLine: true, color: C.body, fontSize: 11 } },
      { text: "", options: { breakLine: true, fontSize: 6 } },
      { text: "领域局限性", options: { breakLine: true, bold: true, color: C.title, fontSize: 12 } },
      { text: "仅在鸟类识别场景验证", options: { color: C.body, fontSize: 11 } },
    ], { x: 0.8, y: 1.7, w: 3.7, h: 2.5, fontFace: "Calibri", valign: "top" });

    // Future
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 5.2, y: 1.2, w: 4.3, h: 3.2, fill: { color: "F0FFF4" },
      line: { color: "C6F6D5", width: 1 }, shadow: makeShadow(),
    });
    slide.addText("未来方向", {
      x: 5.4, y: 1.3, w: 3.9, h: 0.35, fontSize: 16, fontFace: "Calibri",
      color: C.green, bold: true,
    });
    slide.addText([
      { text: "Label-free CBM", options: { breakLine: true, bold: true, color: C.title, fontSize: 12 } },
      { text: "自动发现有意义的概念，无需人工标注", options: { breakLine: true, color: C.body, fontSize: 11 } },
      { text: "", options: { breakLine: true, fontSize: 6 } },
      { text: "概念级对抗鲁棒性", options: { breakLine: true, bold: true, color: C.title, fontSize: 12 } },
      { text: "研究对概念层的对抗攻击", options: { breakLine: true, color: C.body, fontSize: 11 } },
      { text: "", options: { breakLine: true, fontSize: 6 } },
      { text: "跨领域扩展", options: { breakLine: true, bold: true, color: C.title, fontSize: 12 } },
      { text: "医疗影像、自动驾驶等领域", options: { color: C.body, fontSize: 11 } },
    ], { x: 5.5, y: 1.7, w: 3.7, h: 2.5, fontFace: "Calibri", valign: "top" });
  }

  // ── Slide 16: Summary ──────────────────────────────────────────────
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bg };
    slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
    slide.addText("总结", {
      x: 0.8, y: 0.3, w: 8.4, h: 0.7, fontSize: 32, fontFace: "Georgia",
      color: C.title, bold: true,
    });

    const points = [
      { icon: icons.light, text: "CBM 通过概念瓶颈实现内在可解释性，解释与推理一体化" },
      { icon: icons.balance, text: "准确率代价小 (2-5%)，但解释质量显著提升" },
      { icon: icons.flask, text: "干预实验验证了概念的因果性，而非仅仅是相关性" },
      { icon: icons.check, text: "好的解释 = 忠实 + 完整 + 可操作 + 人类对齐" },
    ];

    points.forEach((p, i) => {
      const y = 1.3 + i * 0.9;
      slide.addImage({ data: p.icon, x: 0.8, y: y + 0.05, w: 0.45, h: 0.45 });
      slide.addText(p.text, {
        x: 1.5, y, w: 7.5, h: 0.6, fontSize: 16, fontFace: "Calibri",
        color: C.body, valign: "middle",
      });
    });

    // Quote box
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 1.5, y: 4.6, w: 7.0, h: 0.6, fill: { color: C.accentLight },
    });
    slide.addText("\"A good explanation is one that lets you understand, verify, and correct the model.\"", {
      x: 1.7, y: 4.6, w: 6.6, h: 0.6, fontSize: 13, fontFace: "Georgia",
      color: C.accent, italic: true, valign: "middle", align: "center",
    });
  }

  // ── Slide 17: References ───────────────────────────────────────────
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bg };
    slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
    slide.addText("参考文献", {
      x: 0.8, y: 0.3, w: 8.4, h: 0.7, fontSize: 26, fontFace: "Georgia",
      color: C.title, bold: true,
    });

    slide.addText([
      { text: "[1] Koh, P.W., Nguyen, T., Tang, Y.S., et al. \"Concept Bottleneck Models.\" ICML 2020.", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "", options: { breakLine: true, fontSize: 6 } },
      { text: "[2] Selvaraju, R.R., et al. \"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.\" ICCV 2017.", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "", options: { breakLine: true, fontSize: 6 } },
      { text: "[3] Wah, C., et al. \"The Caltech-UCSD Birds-200-2011 Dataset.\" Technical Report, 2011.", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "", options: { breakLine: true, fontSize: 6 } },
      { text: "[4] Rudin, C. \"Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead.\" Nature Machine Intelligence, 2019.", options: { breakLine: true, color: C.body, fontSize: 12 } },
      { text: "", options: { breakLine: true, fontSize: 6 } },
      { text: "[5] He, K., et al. \"Deep Residual Learning for Image Recognition.\" CVPR 2016.", options: { color: C.body, fontSize: 12 } },
    ], { x: 0.8, y: 1.2, w: 8.4, h: 4.0, fontFace: "Calibri", valign: "top" });
  }

  // ── Slide 18: Thank You ────────────────────────────────────────────
  {
    const slide = pres.addSlide();
    slide.background = { color: C.bg };
    slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
    slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.35, w: 10, h: 0.28, fill: { color: C.accent } });

    slide.addText("Thank You", {
      x: 1, y: 1.5, w: 8, h: 1.2, fontSize: 48, fontFace: "Georgia",
      color: C.title, bold: true, align: "center", valign: "middle",
    });
    slide.addText("感谢聊听！欢迎提问与讨论", {
      x: 1, y: 2.8, w: 8, h: 0.6, fontSize: 18, fontFace: "Calibri",
      color: C.body, align: "center",
    });
    slide.addText("Q & A", {
      x: 1, y: 3.6, w: 8, h: 0.8, fontSize: 28, fontFace: "Georgia",
      color: C.accent, align: "center",
    });
  }

  // ── Save ───────────────────────────────────────────────────────────
  const outPath = "/Users/macbookair/Library/Mobile Documents/com~apple~CloudDocs/BIT/Curriculum/2025-2026-2/Explainable_artificial_intelligence/Presentation/Internal_explaination/CBM_presentation.pptx";
  await pres.writeFile({ fileName: outPath });
  console.log("PPT saved to: " + outPath);
}

main().catch(err => { console.error(err); process.exit(1); });
