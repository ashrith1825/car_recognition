# Car Recognition - Client-Side Inference with ONNX

This project has been updated to run **entirely in the browser** without needing a backend server! 🎉

## What Changed?

Previously, the app required:
- ✗ Flask backend server running
- ✗ Python installed
- ✗ Server to handle predictions

Now it works with:
- ✅ Pure client-side inference
- ✅ No backend server needed
- ✅ Just deploy static files to Render, Netlify, Vercel, etc.

## How It Works

The PyTorch model (`model.pth`) has been converted to ONNX format (`model.onnx`), which can run directly in the browser using ONNX Runtime Web.

### Files Generated:
- `client/public/model.onnx` - The model weights and architecture (91 MB)
- `client/public/classes.json` - Class mappings for the 196 car models

## Setup & Run

### First Time Setup:
```bash
cd client
npm install
```

### Run Development Server:
```bash
npm run dev
```

Open http://localhost:5173 in your browser!

### Build for Production:
```bash
npm run build
```

This creates a `dist` folder with static files ready to deploy.

## Deploying to Render (Static Site)

1. **Push your code to GitHub**

2. **Create a new Static Site on Render:**
   - Go to https://render.com
   - Click "New +" → "Static Site"
   - Connect your GitHub repository

3. **Configure the deployment:**
   - **Build Command:** `cd client && npm install && npm run build`
   - **Publish Directory:** `client/dist`

4. **Deploy!** 
   - Render will build and deploy your site
   - No backend server needed - it's all static files

## How to Re-Convert the Model (If Needed)

If you retrain your model and get a new `model.pth`, convert it again:

```bash
cd backend
python convert_to_onnx.py
```

This will regenerate:
- `client/public/model.onnx`
- `client/public/classes.json`

## Model Performance

- **Model:** ResNet50
- **Classes:** 196 car models
- **Model Size:** ~91 MB
- **Inference:** Runs in browser using WebAssembly
- **First Load:** ~2-5 seconds (downloads model once)
- **Subsequent Predictions:** < 1 second

## Architecture

```
┌─────────────────────────────────────┐
│         User's Browser              │
│  ┌───────────────────────────────┐  │
│  │   React App (Vite)            │  │
│  │                               │  │
│  │   onnxruntime-web             │  │
│  │   (WebAssembly inference)     │  │
│  │                               │  │
│  │   model.onnx (91 MB)          │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘

No backend server needed! ✨
```

## Benefits of Client-Side Inference

1. **No Server Costs:** Just deploy static files (free on many platforms)
2. **Privacy:** Images never leave the user's browser
3. **Scalability:** Each user runs inference on their own device
4. **Fast:** No network latency after initial model load
5. **Offline-Capable:** Works offline after first load (with PWA setup)

## Supported Browsers

- Chrome/Edge (recommended)
- Firefox
- Safari
- Any browser with WebAssembly support

## Troubleshooting

### Model not loading?
- Check browser console for errors
- Ensure `model.onnx` and `classes.json` are in `public/` folder
- Try clearing browser cache

### Slow inference?
- First inference is slower (model initialization)
- Subsequent predictions are much faster
- Consider quantizing model for smaller size

### Out of memory?
- The model is ~91 MB, which should work on most devices
- Close other tabs to free up memory
- Consider model quantization for smaller size

## Original Backend (Still Available)

The Flask backend is still available in the `backend/` folder if you want to use it:

```bash
cd backend
python app.py
```

But it's no longer needed for the React app!

---

Made with ❤️ using PyTorch, ONNX, React, and ONNX Runtime Web
