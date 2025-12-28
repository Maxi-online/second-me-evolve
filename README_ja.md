![Second Me](https://github.com/mindverse/Second-Me/blob/master/images/cover.png)

<div align="center">
  
[![Homepage](https://img.shields.io/badge/Second_Me-Homepage-blue?style=flat-square&logo=homebridge)](https://www.secondme.io/)
[![Report](https://img.shields.io/badge/Paper-arXiv-red?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2503.08102)
[![Discord](https://img.shields.io/badge/Chat-Discord-5865F2?style=flat-square&logo=discord&logoColor=white)](https://discord.gg/GpWHQNUwrg)
[![Twitter](https://img.shields.io/badge/Follow-@SecondMe_AI-1DA1F2?style=flat-square&logo=x&logoColor=white)](https://x.com/SecondMe_AI1)
[![Reddit](https://img.shields.io/badge/Join-Reddit-FF4500?style=flat-square&logo=reddit&logoColor=white)](https://www.reddit.com/r/SecondMeAI/)

</div>

> Fork notice (Maxim Strekolovskiy): このリポジトリは Mindverse/Second-Me のフォークです。上流は Apache-2.0（`LICENSE`）で提供され、このフォークは追加の改良を含みます（`NOTICE` と `README.md` を参照）。


## 私たちのビジョン

OpenAIのような企業は、人間の独立性を脅かす「スーパーAI」を構築しました。私たちは個性を求めています：あなたを消すのではなく、強化するAI。

私たちは「**Second Me**」でそれに挑戦しています：あなた自身の**AI自己**を作成するオープンソースのプロトタイプです。これは、あなたを保存し、あなたのコンテキストを提供し、あなたの利益を守る新しいAI種です。

これは**ローカルでトレーニングされ、ホストされる**ため、あなたのデータはあなたの管理下にありますが、**グローバルに接続**され、AIネットワーク全体であなたの知性を拡張します。それを超えて、これはあなたのAIアイデンティティインターフェースです。これは、あなたのAIを世界にリンクし、AI自己間のコラボレーションを促進し、明日の真にネイティブなAIアプリを構築するための大胆な標準です。

私たちに参加してください。技術愛好家、AI専門家、ドメインエキスパート—Second Meは、あなたの心をデジタルの地平線に拡張するための出発点です。

## 主な機能

### **AI自己をトレーニング**するAIネイティブメモリ ([論文](https://arxiv.org/abs/2503.08102))
今日からあなたのSecond Meをあなた自身の記憶でトレーニングし始めましょう！階層的メモリモデリング（HMM）とMe-Alignmentアルゴリズムを使用して、あなたのAI自己はあなたのアイデンティティをキャプチャし、あなたのコンテキストを理解し、あなたを本物に反映します。

 <p align="center">
  <img src="https://github.com/user-attachments/assets/a84c6135-26dc-4413-82aa-f4a373c0ff89" width="94%" />
</p>


### **知性を拡張**するSecond Meネットワーク
あなたのAI自己をラップトップから私たちの分散ネットワークに起動します。誰でも、どのアプリでも、あなたの許可を得て接続し、あなたのコンテキストをデジタルアイデンティティとして共有します。

<p align="center">
  <img src="https://github.com/user-attachments/assets/9a74a3f4-d8fd-41c1-8f24-534ed94c842a" width="94%" />
</p>


### 明日のアプリをSecond Meで構築
**ロールプレイ**：あなたのAI自己が異なるシナリオであなたを代表するためにペルソナを切り替えます。  
**AIスペース**：他のSecond Meと協力してアイデアを出し合ったり、問題を解決したりします。

<p align="center">
  <img src="https://github.com/user-attachments/assets/bc6125c1-c84f-4ecc-b620-8932cc408094" width="94%" />
</p>

### 100% **プライバシーとコントロール**
従来の集中型AIシステムとは異なり、Second Meはあなたの情報と知性がローカルに完全にプライベートに保たれることを保証します。



## 始め方と私たちとの連絡を保つ方法
スターを付けて参加し、GitHubからのすべてのリリース通知を遅延なく受け取ることができます！


 <p align="center">
  <img src="https://github.com/user-attachments/assets/5c14d956-f931-4c25-b0b3-3c2c96cd7581" width="94%" />
</p>

## クイックスタート

### インストールとセットアップ

#### 前提条件
- macOSオペレーティングシステム
- Python 3.8以上
- Node.js 16以上（フロントエンド用）
- Xcodeコマンドラインツール

#### Xcodeコマンドラインツールのインストール
まだXcodeコマンドラインツールをインストールしていない場合は、次のコマンドを実行してインストールできます：
```bash
xcode-select --install
```

インストール後、ライセンス契約に同意する必要があるかもしれません：
```bash
sudo xcodebuild -license accept
```

1. リポジトリをクローン
```bash
git clone git@github.com:Mindverse/Second-Me.git
cd Second-Me
```

2. 環境を設定

#### オプションA：既存のconda環境を持つユーザー向け
すでにcondaをインストールしている場合：

1) 環境ファイルから新しい環境を作成：
```bash
conda env create -f environment.yml   # これにより、'second-me'という名前の環境が作成されます
conda activate second-me
```

2) `.env`でカスタムcondaモードを設定：
```bash
CUSTOM_CONDA_MODE=true
```

3) セットアップを実行：
```bash
make setup
```

#### オプションB：新規ユーザー向け
新規ユーザーまたは新しい環境を希望する場合：
```bash
make setup
```

このコマンドは自動的に次のことを行います：
- 必要なすべてのシステム依存関係をインストール（condaが存在しない場合は含む）
- 'second-me'という名前の新しいPython環境を作成
- llama.cppをビルド
- フロントエンド環境を設定

3. サービスを開始
```bash
make start
```

4. サービスにアクセス
ブラウザを開き、`http://localhost:3000`にアクセス

5. ヘルプとその他のコマンドを表示
```bash
make help
```

### 重要な注意事項
1. 十分なディスクスペースを確保してください（少なくとも10GBを推奨）
2. 既存のconda環境を使用する場合、競合するパッケージバージョンがないことを確認してください
3. 初回の起動には依存関係のダウンロードとインストールに数分かかることがあります
4. 一部のコマンドはsudo権限を必要とする場合があります

### トラブルシューティング
問題が発生した場合、次の点を確認してください：
1. PythonとNode.jsのバージョンが要件を満たしていること
2. 正しいconda環境にいること
3. すべての依存関係が適切にインストールされていること
4. システムのファイアウォールがアプリケーションが必要とするポートを使用できるようにしていること

## チュートリアルとユースケース
🛠️ [ユーザーチュートリアル](https://second-me.gitbook.io/a-new-ai-species-making-we-matter-again)に従って、あなたのSecond Meを構築してください。

💡 以下のリンクをチェックして、Second Meが実際のシナリオでどのように使用されるかを確認してください：
- [Felix AMA（ロールプレイアプリ）](https://app.secondme.io/example/ama)
- [15日間のヨーロッパ都市旅程をブレインストーミング（ネットワークアプリ）](https://app.secondme.io/example/brainstorming)
- [スピードデートのマッチとしてのアイスブレイク（ネットワークアプリ）](https://app.secondme.io/example/Icebreaker)

## コミュニティに参加
- [Discord](https://discord.com/invite/GpWHQNUwrg)
- [Reddit](https://www.reddit.com/r/SecondMeAI/)
- [X](https://x.com/SecondMe_AI1)

## 近日公開

以下の機能は内部で完了しており、オープンソースプロジェクトに段階的に統合されています。詳細な実験結果と技術仕様については、[技術報告書](https://arxiv.org/abs/2503.08102)を参照してください。

### モデル強化機能
- [ ] **長い連鎖思考トレーニングパイプライン**：拡張された思考プロセストレーニングによる強化された推論能力
- [ ] **L2モデルの直接選好最適化**：ユーザーの選好と意図に対する改善された整合性
- [ ] **トレーニング用データフィルタリング**：高品質なトレーニングデータ選択のための高度な技術
- [ ] **Apple Siliconサポート**：MLXトレーニングおよびサービング機能を備えたApple Siliconプロセッサのネイティブサポート

### 製品機能
- [ ] **自然言語メモリ要約**：自然言語形式での直感的なメモリ整理


## 貢献

Second Meへの貢献を歓迎します！バグの修正、新機能の追加、ドキュメントの改善に興味がある場合は、貢献ガイドを確認してください。また、Second Meをコミュニティ、技術会議、またはソーシャルメディアで共有することでSecond Meをサポートすることもできます。

開発に関する詳細な情報については、[貢献ガイド](./CONTRIBUTING.md)を参照してください。

## 貢献者

Second Meに貢献してくれたすべての個人に感謝の意を表します！知性のアップロードの未来に貢献することに興味がある場合、コード、ドキュメント、アイデアを通じて、私たちのリポジトリにプルリクエストを提出してください：[Second-Me](https://github.com/Mindverse/Second-Me)。


<a href="https://github.com/mindverse/Second-Me/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=mindverse/Second-Me" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

## 謝辞

この作業はオープンソースコミュニティの力を活用しています。 

データ合成には、Microsoftの[GraphRAG](https://github.com/microsoft/graphrag)を使用しました。

モデルのデプロイには、効率的な推論機能を提供する[llama.cpp](https://github.com/ggml-org/llama.cpp)を使用しました。

私たちのベースモデルは主に[Qwen2.5](https://huggingface.co/Qwen)シリーズから来ています。

また、Second Meを体験してくれたすべてのユーザーに心から感謝します。パイプライン全体で最適化の余地が大いにあることを認識しており、皆さんがローカルで最高の体験を楽しめるようにするために、継続的な改善に全力を尽くします。

## ライセンス

Second MeはApache License 2.0の下でライセンスされたオープンソースソフトウェアです。詳細については、[LICENSE](LICENSE)ファイルを参照してください。

[license]: ./LICENSE

## Star History

<a href="https://www.star-history.com/#mindverse/Second-Me&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=mindverse/Second-Me&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=mindverse/Second-Me&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=mindverse/Second-Me&type=Date" />
 </picture>
</a>
