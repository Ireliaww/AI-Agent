#!/bin/bash

# Google Cloud Run 部署脚本
# 使用方法: ./deploy-cloud-run.sh

set -e

echo "🚀 开始部署到 Google Cloud Run"
echo ""

# 配置变量
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"your-project-id"}
SERVICE_NAME="multi-agent-ai"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# 检查gcloud是否已安装
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI未安装"
    echo "请访问: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# 提示用户设置项目ID（如果未设置）
if [ "$PROJECT_ID" = "your-project-id" ]; then
    echo "请设置您的Google Cloud项目ID:"
    read -r PROJECT_ID
    echo ""
fi

# 设置项目
echo "📋 设置项目: $PROJECT_ID"
gcloud config set project "$PROJECT_ID"
echo ""

# 启用必要的API
echo "🔧 启用Cloud Run和Container Registry API..."
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
echo ""

# 构建Docker镜像
echo "🐳 构建Docker镜像..."
gcloud builds submit --tag "$IMAGE_NAME"
echo ""

# 检查GOOGLE_API_KEY
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "⚠️  GOOGLE_API_KEY环境变量未设置"
    echo "请输入您的Google API Key:"
    read -r GOOGLE_API_KEY
    echo ""
fi

# 部署到Cloud Run
echo "🚀 部署到Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
    --image "$IMAGE_NAME" \
    --platform managed \
    --region "$REGION" \
    --allow-unauthenticated \
    --set-env-vars "GOOGLE_API_KEY=$GOOGLE_API_KEY" \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --min-instances 0

echo ""
echo "✅ 部署完成！"
echo ""
echo "您的服务URL:"
gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format="value(status.url)"
echo ""
echo "📚 API文档: \$(gcloud run services describe $SERVICE_NAME --region $REGION --format='value(status.url)')/docs"
