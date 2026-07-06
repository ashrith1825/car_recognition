import React, { useState, useRef, useEffect } from 'react';
import * as ort from 'onnxruntime-web';

// SVG icons as components
const UploadIcon = () => (
  <svg
    className="upload-icon"
    stroke="currentColor"
    fill="none"
    viewBox="0 0 48 48"
    aria-hidden="true"
  >
    <path
      d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const Spinner = ({ buttonSpinner = false }) => <div className={`spinner ${buttonSpinner ? 'button-spinner' : ''}`}></div>;

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [modelLoading, setModelLoading] = useState(true);
  const [modelSession, setModelSession] = useState(null);
  const [classes, setClasses] = useState(null);
  const fileInputRef = useRef(null);

  // Load ONNX model on component mount
  useEffect(() => {
    const loadModel = async () => {
      try {
        console.log('🔄 Loading ONNX model...');
        
        // Load the model
        const session = await ort.InferenceSession.create('/model.onnx', {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all'
        });
        
        setModelSession(session);
        console.log('✅ Model loaded successfully!');

        // Load class mappings
        const classesResponse = await fetch('/classes.json');
        const classesData = await classesResponse.json();
        setClasses(classesData);
        console.log('✅ Class mappings loaded!');

        setModelLoading(false);
      } catch (err) {
        console.error('❌ Failed to load model:', err);
        setError('Failed to load the model. Please refresh the page.');
        setModelLoading(false);
      }
    };

    loadModel();
  }, []);

  const preprocessImage = async (imageFile) => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');

      img.onload = () => {
        // Resize to 224x224
        canvas.width = 224;
        canvas.height = 224;
        ctx.drawImage(img, 0, 0, 224, 224);

        // Get image data
        const imageData = ctx.getImageData(0, 0, 224, 224);
        const { data } = imageData;

        // Normalize using ImageNet mean and std
        const mean = [0.485, 0.456, 0.406];
        const std = [0.229, 0.224, 0.225];
        
        // Convert to tensor format [1, 3, 224, 224]
        const float32Data = new Float32Array(1 * 3 * 224 * 224);
        
        for (let i = 0; i < 224; i++) {
          for (let j = 0; j < 224; j++) {
            const pixelIndex = i * 224 + j;
            const dataIndex = pixelIndex * 4;
            
            // Red channel
            float32Data[pixelIndex] = (data[dataIndex] / 255 - mean[0]) / std[0];
            // Green channel
            float32Data[224 * 224 + pixelIndex] = (data[dataIndex + 1] / 255 - mean[1]) / std[1];
            // Blue channel
            float32Data[2 * 224 * 224 + pixelIndex] = (data[dataIndex + 2] / 255 - mean[2]) / std[2];
          }
        }

        resolve(float32Data);
      };

      img.onerror = reject;
      img.src = URL.createObjectURL(imageFile);
    });
  };

  const softmax = (arr) => {
    const max = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPrediction(null);
      setError('');
    } else {
      setSelectedFile(null);
      setPreviewUrl(null);
      setError('Please select a valid image file (PNG, JPG).');
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please select an image first.');
      return;
    }

    if (!modelSession || !classes) {
      setError('Model is still loading. Please wait...');
      return;
    }

    setIsLoading(true);
    setPrediction(null);
    setError('');

    try {
      console.log('🔄 Preprocessing image...');
      const inputData = await preprocessImage(selectedFile);
      
      console.log('🔄 Running inference...');
      const tensor = new ort.Tensor('float32', inputData, [1, 3, 224, 224]);
      const feeds = { input: tensor };
      const results = await modelSession.run(feeds);
      
      // Get output tensor
      const output = results.output.data;
      
      // Apply softmax to get probabilities
      const probabilities = softmax(Array.from(output));
      
      // Get top prediction
      const maxIndex = probabilities.indexOf(Math.max(...probabilities));
      const confidence = probabilities[maxIndex];
      
      // Get class name
      const carName = classes.idx_to_class[maxIndex.toString()];
      
      console.log('✅ Prediction complete!');
      setPrediction({
        carName: carName,
        confidence: confidence,
      });
    } catch (err) {
      console.error('❌ Prediction failed:', err);
      setError(err.message || 'Failed to make prediction.');
    } finally {
      setIsLoading(false);
    }
  };

  const triggerFileSelect = () => {
    fileInputRef.current.click();
  };
  
  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setPrediction(null);
    setError('');
    setIsLoading(false);
    if(fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="app-container">
      <style>{`
        :root {
          --background-color: #ECECEC; 
          --content-background-color: #F9F9F9;
          --primary-text-color: #1A1A1A; 
          --secondary-text-color: #444444; 
          --border-color: #C8C8C8;
          --button-color: #357A94; 
          --button-hover-color: #2C667B;
          --highlight-color: #E89C75; 
          --error-color: #9B1B1B;
          --error-background-color: rgba(155, 27, 27, 0.1);
          --font-family: 'Inter', sans-serif;
        }

        body {
          margin: 0;
          font-family: var(--font-family);
          background-color: var(--background-color);
          color: var(--primary-text-color);
        }

        .app-container {
          display: flex;
          justify-content: center;
          align-items: flex-start;
          min-height: 100vh;
          width: 100%;
          padding: 4rem 2rem;
          box-sizing: border-box;
        }

        .content-wrapper {
          width: 100%;
          max-width: 1100px;
          display: grid;
          grid-template-columns: 1fr;
          gap: 2rem;
          align-items: flex-start;
        }

        @media (min-width: 1024px) {
            .content-wrapper {
                grid-template-columns: 1fr 1fr;
            }
        }
        
        .main-content {
          background-color: var(--content-background-color);
          border-radius: 1rem;
          padding: 2.5rem;
          box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.07);
        }

        .results-panel {
          position: sticky;
          top: 4rem;
        }
        
        .header {
          text-align: center;
          margin-bottom: 2rem;
        }
        .title {
          font-size: 2.25rem;
          font-weight: bold;
          color: var(--primary-text-color);
        }
        .subtitle {
          margin-top: 0.5rem;
          color: var(--secondary-text-color);
        }

        .model-status {
          text-align: center;
          padding: 1rem;
          margin-bottom: 1rem;
          border-radius: 0.5rem;
          font-size: 0.9rem;
        }

        .model-status.loading {
          background-color: #FFF4E6;
          color: #E67700;
        }

        .model-status.ready {
          background-color: #E6F7E6;
          color: #2E7D32;
        }

        .upload-area {
          border: 2px dashed var(--border-color);
          border-radius: 0.75rem;
          padding: 1.5rem;
          text-align: center;
          cursor: pointer;
          display: flex;
          flex-direction: column;
          justify-content: center;
          transition: all 0.3s ease-in-out;
          min-height: 200px;
        }
        .upload-area:hover {
          border-color: var(--button-color);
          background-color: rgba(53, 122, 148, 0.07);
        }
        .upload-area.has-preview {
          padding: 1rem;
          min-height: auto;
        }
        .file-input {
          display: none;
        }
        .image-preview {
          max-height: 300px;
          width: 100%;
          object-fit: cover;
          border-radius: 0.5rem;
        }
        .upload-prompt .upload-icon {
          width: 3rem;
          height: 3rem;
          margin: 0 auto 0.5rem;
          color: var(--secondary-text-color);
        }
        .upload-prompt .upload-link {
          font-weight: 600;
          color: var(--button-color);
        }
        .upload-prompt .upload-hint {
          font-size: 0.75rem;
          color: #9E9E9E;
          margin-top: 0.25rem;
        }

        .file-info {
          text-align: center;
          margin-top: 1rem;
          font-size: 0.875rem;
          color: var(--secondary-text-color);
        }
        .file-info span {
          font-weight: 500;
          color: var(--primary-text-color);
        }

        .button-container {
          margin-top: 2rem;
        }
        
        .action-buttons-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 1rem;
        }

        .predict-button, .change-image-button {
          width: 100%;
          font-weight: bold;
          font-size: 1.125rem;
          padding: 1rem;
          border-radius: 0.5rem;
          cursor: pointer;
          display: flex;
          justify-content: center;
          align-items: center;
          transition: all 0.3s ease-in-out;
        }

        .predict-button {
          background-color: var(--button-color);
          color: white;
          border: none;
        }

        .predict-button:hover:not(:disabled) {
          background-color: var(--button-hover-color);
          box-shadow: 0 4px 14px rgba(53, 122, 148, 0.4);
        }
        .predict-button:disabled {
          background-color: #D9D9D9;
          color: #A1A1A1;
          cursor: not-allowed;
        }

        .change-image-button {
          background-color: transparent;
          color: var(--button-color);
          border: 2px solid var(--button-color);
        }
        
        .change-image-button:hover:not(:disabled) {
            background-color: rgba(53, 122, 148, 0.07);
        }
        
        .change-image-button:disabled {
            color: #A1A1A1;
            border-color: #D9D9D9;
            cursor: not-allowed;
        }

        .spinner {
          width: 48px;
          height: 48px;
          border: 5px solid var(--border-color);
          border-top-color: var(--button-color);
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }

        .spinner.button-spinner {
            width: 28px;
            height: 28px;
            border: 4px solid rgba(255, 255, 255, 0.5);
            border-top-color: white;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .error-box, .result-card, .placeholder-card {
          padding: 1.5rem 2rem;
          border-radius: 1rem;
          animation: fadeIn 0.5s ease-in-out forwards;
          background-color: var(--content-background-color);
          box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.07);
        }

        .error-box {
          background-color: var(--error-background-color);
          border: 1px solid var(--error-color);
          color: var(--error-color);
          text-align: center;
        }
        .result-card {
          background-color: #F4F4F4;
          border: 1px solid var(--border-color);
        }

        .placeholder-card {
            border: 2px dashed var(--border-color);
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 400px;
            color: var(--secondary-text-color);
            box-shadow: none;
            background-color: transparent;
        }

        .placeholder-card h2 {
            font-size: 1.5rem;
            color: var(--primary-text-color);
            margin-bottom: 0.5rem;
        }

        .result-title {
          text-align: center;
          font-size: 1.5rem;
          font-weight: 600;
          margin-bottom: 1rem;
        }
        .result-item {
          display: flex;
          justify-content: space-between;
          margin-top: 0.5rem;
          font-size: 1.125rem;
        }
        .result-item span:first-child {
          color: var(--secondary-text-color);
        }
        .result-item .result-value {
          font-weight: bold;
          color: var(--primary-text-color);
        }

        .confidence-bar-container {
          width: 100%;
          background-color: var(--border-color);
          border-radius: 999px;
          height: 0.65rem;
          margin-top: 1rem;
          overflow: hidden;
        }
        .confidence-bar {
          background-color: var(--highlight-color);
          height: 100%;
          border-radius: 999px;
          transition: width 0.5s ease-in-out;
        }
        .confidence-text {
          text-align: right;
          color: var(--secondary-text-color);
          margin-top: 0.5rem;
          font-size: 0.875rem;
        }

        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>

      <div className="content-wrapper">
        <div className="main-content">
            <div className="header">
              <h1 className="title">Car Recognition AI</h1>
              <p className="subtitle">Upload an image to identify the car model.</p>
            </div>

            {modelLoading && (
              <div className="model-status loading">
                ⏳ Loading AI model... Please wait
              </div>
            )}

            {!modelLoading && modelSession && (
              <div className="model-status ready">
                ✅ Model ready! Running entirely in your browser - no server needed
              </div>
            )}

            <div
              className={`upload-area ${previewUrl ? 'has-preview' : ''}`}
              onClick={!selectedFile ? triggerFileSelect : undefined}
              role="button"
              tabIndex="0"
              aria-label="Upload image area"
            >
              <input
                ref={fileInputRef}
                type="file"
                className="file-input"
                accept="image/*"
                onChange={handleFileChange}
              />
              {previewUrl ? (
                <img src={previewUrl} alt="Selected car" className="image-preview" />
              ) : (
                <div className="upload-prompt">
                  <UploadIcon />
                  <p>
                    <span className="upload-link">Click to upload</span> or drag and drop
                  </p>
                  <p className="upload-hint">PNG, JPG, JPEG up to 10MB</p>
                </div>
              )}
            </div>

            {selectedFile && !prediction && !error && (
              <div className="file-info">
                File: <span>{selectedFile.name}</span>
              </div>
            )}

            <div className="button-container">
                {prediction || error ? (
                    <button onClick={handleReset} className="predict-button">
                        Upload Another Image
                    </button>
                ) : selectedFile ? (
                    <div className="action-buttons-grid">
                        <button
                            onClick={triggerFileSelect}
                            disabled={isLoading || modelLoading}
                            className="change-image-button"
                        >
                            Change Image
                        </button>
                        <button
                            onClick={handlePredict}
                            disabled={isLoading || modelLoading}
                            className="predict-button"
                        >
                            {isLoading ? <Spinner buttonSpinner={true} /> : 'Identify Car'}
                        </button>
                    </div>
                ) : null }
            </div>
        </div>

        <div className="results-panel">
            {isLoading ? (
                <div className="placeholder-card">
                    <Spinner />
                    <p style={{marginTop: '1rem', color: 'var(--primary-text-color)', fontSize: '1.125rem'}}>Analyzing image...</p>
                </div>
            ) : error ? (
                <div className="error-box">
                    <p>{error}</p>
                </div>
            ) : prediction ? (
                <div className="result-card">
                  <h2 className="result-title">Prediction Result</h2>
                  <div className="result-item">
                    <span>Car Model:</span>
                    <span className="result-value">{prediction.carName}</span>
                  </div>
                  <div className="confidence-bar-container">
                    <div
                      className="confidence-bar"
                      style={{ width: `${prediction.confidence * 100}%` }}
                    ></div>
                  </div>
                  <div className="confidence-text">
                    Confidence: {(prediction.confidence * 100).toFixed(2)}%
                  </div>
                </div>
            ) : (
                <div className="placeholder-card">
                    <h2>Results Panel</h2>
                    <p>Your prediction will appear here after uploading an image.</p>
                </div>
            )}
        </div>
      </div>
    </div>
  );
}

