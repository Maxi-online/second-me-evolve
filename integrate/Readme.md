# Second-Me WeChat Bot 集成指南

> Fork notice (Maxim Strekolovskiy): 本仓库是 Second Me (Mindverse) 的衍生/分支版本。上游代码遵循 Apache-2.0（见 `LICENSE`），本分支包含额外改动（见 `NOTICE`）。

这个项目将 Second-Me 与微信机器人进行集成，让您可以通过微信与您的 Second-Me 进行交互。

## 功能特点

- 🤖 将 Second-Me 无缝接入微信
- 💬 支持文本消息的智能回复
- 🔄 自动化的消息处理流程
- 📝 完整的日志记录
- ⚠️ 健壮的错误处理机制

## 安装要求

- Python 3.8+
- WeChat 个人账号
- Second-Me 项目环境

## 快速开始

1. 克隆项目并安装依赖：

```bash
git clone https://github.com/mindverse/second-me.git
cd second-me/integrate
pip install -r requirements.txt
```

2. 配置环境：
   - 确保 Second-Me 的配置文件正确设置
   - 检查 `.env` 文件中的必要配置

3. 运行机器人：

```bash
python wechat_bot.py
```

4. 首次运行时，扫描终端显示的二维码登录微信

## 项目结构
