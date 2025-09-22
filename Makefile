.PHONY: clean build test lint format check install publish publish-test help

# パッケージ名とバージョンを取得
PACKAGE_NAME := $(shell poetry version | cut -d' ' -f1)
PACKAGE_VERSION := $(shell poetry version -s)

help: ## このヘルプメッセージを表示
	@echo "利用可能なコマンド:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

clean: ## ビルド成果物をクリーンアップ
	@echo "🧹 クリーンアップ中..."
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

format: ## コードをフォーマット
	@echo "🎨 コードフォーマット中..."
	poetry run black .
	poetry run isort .

lint: ## コードをリント
	@echo "🔍 リント中..."
	poetry run flake8 .
	poetry run mypy .

test: ## テストを実行
	@echo "🧪 テスト実行中..."
	poetry run pytest

check: format lint test ## フォーマット、リント、テストを実行

install: ## 依存関係をインストール
	@echo "📦 依存関係インストール中..."
	poetry install

build: clean ## パッケージをビルド
	@echo "🔨 パッケージビルド中..."
	poetry build
	@echo "✅ ビルド完了: dist/$(PACKAGE_NAME)-$(PACKAGE_VERSION).tar.gz"
	@echo "✅ ビルド完了: dist/$(PACKAGE_NAME)-$(PACKAGE_VERSION)-py3-none-any.whl"

publish-test: build ## TestPyPIに公開
	@echo "🚀 TestPyPIに公開中..."
	@echo "注意: poetry config repositories.testpypi https://test.pypi.org/legacy/ を事前に実行してください"
	poetry publish -r testpypi
	@echo "✅ TestPyPIに公開完了"
	@echo "インストール確認: pip install --index-url https://test.pypi.org/simple/ $(PACKAGE_NAME)"

publish: build ## PyPIに公開
	@echo "🚀 PyPIに公開中..."
	@read -p "本当にPyPIに公開しますか？ (y/N): " confirm && [ "$$confirm" = "y" ]
	poetry publish
	@echo "✅ PyPIに公開完了"
	@echo "インストール確認: pip install $(PACKAGE_NAME)"

version-patch: ## パッチバージョンを上げる
	@echo "📝 パッチバージョンアップ中..."
	poetry version patch
	@echo "新しいバージョン: $(shell poetry version -s)"

version-minor: ## マイナーバージョンを上げる
	@echo "📝 マイナーバージョンアップ中..."
	poetry version minor
	@echo "新しいバージョン: $(shell poetry version -s)"

version-major: ## メジャーバージョンを上げる
	@echo "📝 メジャーバージョンアップ中..."
	poetry version major
	@echo "新しいバージョン: $(shell poetry version -s)"

info: ## プロジェクト情報を表示
	@echo "📋 プロジェクト情報:"
	@echo "  パッケージ名: $(PACKAGE_NAME)"
	@echo "  バージョン: $(PACKAGE_VERSION)"
	@echo "  Pythonバージョン: $(shell python --version)"
	@echo "  Poetryバージョン: $(shell poetry --version)"

# 開発環境のセットアップ
dev-setup: install ## 開発環境をセットアップ
	@echo "🛠️ 開発環境セットアップ中..."
	poetry run pre-commit install 2>/dev/null || echo "pre-commitが見つかりません（オプション）"
	@echo "✅ 開発環境セットアップ完了"

# 完全なリリースプロセス
release-patch: version-patch check publish ## パッチリリース（テスト→ビルド→公開）

release-minor: version-minor check publish ## マイナーリリース（テスト→ビルド→公開）

release-major: version-major check publish ## メジャーリリース（テスト→ビルド→公開）

# 安全なテストリリース
test-release: check publish-test ## テストリリース（TestPyPIに公開）
