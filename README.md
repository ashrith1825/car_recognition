# Car Recognition - Client-Side Inference with ONNX

This project runs entirely in the browser without needing a backend server.

## What Changed?

Previously, the app required:
- Flask backend server running
- Python installed
- Server-side prediction handling

Now it works with:
- Pure client-side inference
- No backend server needed
- Static deployment to Render, Netlify, Vercel, and similar platforms

## How It Works

The PyTorch model model.pth has been converted to ONNX format, model.onnx, which runs directly in the browser using ONNX Runtime Web.

Files generated:
- client/public/model.onnx - the model weights and architecture
- client/public/classes.json - class mappings for the 196 car models

## Setup & Run

### First Time Setup

```bash
cd client
npm install
```

### Run Development Server

```bash
npm run dev
```

Open http://localhost:5173 in your browser.

### Build for Production

```bash
npm run build
```

This creates a dist folder with static files ready to deploy.

## Deploying to Render

1. Push your code to GitHub.
2. Create a new Static Site on Render.
3. Connect this GitHub repository.
4. Use the build command `cd client && npm install && npm run build`.
5. Set the publish directory to `client/dist`.

## Re-Convert the Model

If you retrain the model and get a new model.pth, convert it again:

```bash
cd backend
python convert_to_onnx.py
```

This regenerates client/public/model.onnx and client/public/classes.json.

## Model Performance

- Model: ResNet50
- Classes: 196 car models
- Model size: about 91 MB
- Inference: browser-based with WebAssembly

## Original Backend

The Flask backend is still available in backend/ if you want to use it:

```bash
cd backend
python app.py
```

It is no longer required for the React app.
"# car_recognition" 
