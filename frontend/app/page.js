'use client';

import { useState } from 'react';

const modelConfig = {
  tfidf: [
    { value: 'lr', label: 'Logistic Regression' },
    { value: 'svm', label: 'SVM' },
    { value: 'rf', label: 'Random Forest' },
    { value: 'nb', label: 'Naive Bayes' },
    { value: 'knn', label: 'KNN' },
  ],
  bow: [
    { value: 'mlp', label: 'MLP' },
    { value: 'lr', label: 'Logistic Regression' },
    { value: 'svm', label: 'SVM' },
    { value: 'rf', label: 'Random Forest' },
    { value: 'nb', label: 'Naive Bayes' },
    { value: 'knn', label: 'KNN' },
  ],
  rnn: [
    { value: 'rnn_lstm', label: 'Bidirectional LSTM (RNN)' }
  ]
};

export default function Home() {
  const [text, setText] = useState('');
  const [featureType, setFeatureType] = useState('tfidf');
  const [modelName, setModelName] = useState('lr');
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFeatureChange = (e) => {
    const newFeatureType = e.target.value;
    setFeatureType(newFeatureType);
    setModelName(modelConfig[newFeatureType][0].value);
    setPrediction(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim()) {
      setError('Vui lòng nhập nội dung tin tức.');
      return;
    }
    setError('');
    setIsLoading(true);
    setPrediction(null);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text, feature_type: featureType, model_name: modelName }),
      });

      if (!response.ok) {
        throw new Error('Lỗi từ server, vui lòng thử lại.');
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8 bg-[url('/background.png')] bg-cover bg-center">
      <div className="w-full max-w-2xl bg-white p-8 rounded-lg shadow-md border border-gray-200">
        <h1 className="text-3xl font-bold text-center mb-2 text-gray-800">
          Dự đoán chủ đề tin tức
        </h1>
        <p className="text-center text-gray-600 mb-6">
          Nhập nội dung, chọn mô hình và cách trích xuất đặc trưng để dự đoán.
        </p>

        <form onSubmit={handleSubmit} className="space-y-6">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Nhập tiêu đề hoặc nội dung tin tức ở đây..."
            className="w-full h-40 p-4 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-shadow duration-200"
            required
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label htmlFor="featureType" className="block text-sm font-medium text-gray-700 mb-1">
                Cách trích xuất đặc trưng
              </label>
              <select
                id="featureType"
                value={featureType}
                onChange={handleFeatureChange}
                className="w-full p-3 border border-gray-300 rounded-md bg-white focus:ring-2 focus:ring-blue-500"
              >
                <option value="tfidf">TF-IDF</option>
                <option value="bow">Bag-of-Words</option>
                <option value="rnn">Deep Learning (RNN)</option>
              </select>
            </div>
            <div>
              <label htmlFor="modelName" className="block text-sm font-medium text-gray-700 mb-1">
                Model
              </label>
              <select
                id="modelName"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-md bg-white focus:ring-2 focus:ring-blue-500"
              >
                {modelConfig[featureType].map((model) => (
                  <option key={model.value} value={model.value}>
                    {model.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full bg-blue-600 text-white font-bold py-3 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-400 transition-all duration-300 ease-in-out transform hover:scale-105"
          >
            {isLoading ? 'Đang dự đoán...' : 'Dự đoán'}
          </button>
        </form>

        {error && <p className="mt-4 text-center text-red-500 font-medium">{error}</p>}

        {prediction && (
          <div className="mt-8 p-6 bg-green-50 border border-green-200 rounded-md text-center transition-opacity duration-500">
            <h2 className="text-xl font-semibold text-gray-800">Kết quả dự đoán</h2>
            <p className="text-3xl font-bold text-green-600 mt-2 capitalize">
              {prediction.category}
            </p>
            {/* {prediction.confidence > 0 && (
                 <p className="text-md text-gray-500 mt-1">
                 Độ tin cậy: { (prediction.confidence * 100).toFixed(2) }%
               </p>
            )} */}
          </div>
        )}
      </div>
      <footer className="w-full text-center text-black-500 text-sm py-4">
        <p>
          Được tạo bởi B2105973 &copy; {new Date().getFullYear()}
        </p>
      </footer>
    </main>
  );
}
