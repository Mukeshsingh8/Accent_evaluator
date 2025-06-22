# 🚀 Free Deployment Guide

## **Streamlit Cloud (Recommended - Completely Free)**

### **Step 1: Create GitHub Repository**

1. Go to [GitHub](https://github.com) and create a new repository
2. Name it `accent-evaluator` or similar
3. Make it public (required for free Streamlit Cloud)

### **Step 2: Push Your Code**

```bash
# Add your GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/accent-evaluator.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### **Step 3: Deploy on Streamlit Cloud**

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Configure your app:
   - **Repository**: `YOUR_USERNAME/accent-evaluator`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: Leave default or customize
5. Click **"Deploy"**

### **Step 4: Configure Secrets**

1. In your Streamlit Cloud dashboard, go to **Settings** → **Secrets**
2. Add your OpenAI API key:
   ```toml
   OPENAI_API_KEY = "your-openai-api-key-here"
   ```
3. Click **"Save"**

### **Step 5: Your App is Live!**

Your app will be available at: `https://YOUR_APP_NAME-YOUR_USERNAME.streamlit.app`

---

## **Alternative Free Platforms**

### **Railway (Free Tier)**

1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. Click **"New Project"** → **"Deploy from GitHub repo"**
4. Select your repository
5. Add environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
6. Deploy

### **Render (Free Tier)**

1. Go to [render.com](https://render.com)
2. Sign in with GitHub
3. Click **"New"** → **"Web Service"**
4. Connect your repository
5. Configure:
   - **Name**: `accent-evaluator`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
6. Add environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
7. Deploy

### **Heroku (Free Tier - Limited)**

1. Create a `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Install Heroku CLI and deploy:
   ```bash
   heroku create your-app-name
   heroku config:set OPENAI_API_KEY=your-api-key
   git push heroku main
   ```

---

## **Important Notes**

### **Free Tier Limitations**

- **Streamlit Cloud**: 
  - ✅ Completely free
  - ✅ Automatic deployments
  - ✅ Custom domains
  - ❌ Limited to 1GB RAM
  - ❌ Sleeps after inactivity

- **Railway**:
  - ✅ $5 free credit monthly
  - ✅ Good performance
  - ❌ Requires credit card

- **Render**:
  - ✅ Free tier available
  - ✅ Automatic deployments
  - ❌ Sleeps after inactivity
  - ❌ Limited bandwidth

### **Environment Variables**

All platforms require these environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key

### **System Dependencies**

The `packages.txt` file ensures FFmpeg is installed on Streamlit Cloud.

### **Troubleshooting**

1. **App won't start**: Check logs in the platform dashboard
2. **API errors**: Verify your OpenAI API key is correct
3. **Audio processing fails**: Ensure FFmpeg is available (handled by `packages.txt`)

---

## **Next Steps**

1. **Monitor Usage**: Check your OpenAI API usage to manage costs
2. **Custom Domain**: Add a custom domain for professional appearance
3. **Analytics**: Add Google Analytics or similar for usage tracking
4. **Backup**: Consider setting up automated backups

---

**🎉 Congratulations! Your English Accent Evaluator is now live and accessible to users worldwide!** 