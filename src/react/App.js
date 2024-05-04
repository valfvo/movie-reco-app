// App.js
import { React, useState } from 'react';
import ImageUpload from './ImageUpload';
import "react-responsive-carousel/lib/styles/carousel.min.css";
import './App.css';

const App = () => {
  const [selectedModel, setSelectedModel] = useState('mobilenet');
  const [inputData, setInputData] = useState(null);
  const [inputDataType, setInputDataType] = useState(null);
  const [recos, setRecos] = useState([]);

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  const handleImageState = (image) => {
    const is_image = typeof image === 'string' && image.startsWith('data:image');
    if (is_image) {
      setInputData(image);
      setInputDataType('poster');
    } else {
      setInputData(null);
      setInputDataType(null);
    }
  }

  const handlePlotChange = (event) => {
    const plot = event.target.value;
    setInputData(plot);
    setInputDataType('plot');
  }

  const handleRecommendations = async () => {
    const query = {
      model: selectedModel,
      data: inputData,
      dtype: inputDataType,
      limit: 6,
    };

    const response = await fetch('http://localhost:8058/recommendation', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(query),
    });

    if (response.ok) {
      const r = await response.json();
      console.log(r);
      setRecos(r['recommendation']);
    } else {
      console.log(response);
    }
  }

  return (
    <div className="app">
      <div className="search-bar-container">
        <input className="search-bar" placeholder="Search a movie..." />
      </div>

      <div className="pt-10 pb-10">
        <input type="radio" value="mobilenet" name="model" onChange={handleModelChange} defaultChecked />
        <label className="pe-6 ps-2">MobileNet</label>
        <input type="radio" value="bert" name="model" onChange={handleModelChange} />
        <label className="pe-6 ps-2">Bert</label>
        <input type="radio" value="count-vectorizer" name="model" onChange={handleModelChange} />
        <label className="pe-6 ps-2">CountVectorizer</label>
      </div>

      <div className="flex space-x-8 justify-center">
        { selectedModel === 'mobilenet' ? (
          <ImageUpload onStateChange={handleImageState}/>
        ) : (
          <textarea className="bg-black border border-white rounded-md px-2.5 py-2.5 h-[32rem] w-96" placeholder="Write a plot..." onChange={handlePlotChange} />
        )}
      </div>

      { inputData && <button className="bg-teal-600 p-3 rounded-full font-bold mt-10" onClick={handleRecommendations}>Get recommendations</button> }

      <div className="grid grid-cols-3 gap-4 mt-20 mx-6 mb-4">
        {recos.map(reco => (
          <div className="grid grid-cols-10 bg-orange-50 text-black rounded-lg overflow-hidden">
            <img src={reco.poster} alt='No poster' className="col-span-4 h-full" />
            <div className="text-left col-span-6 p-4">
              <span className='font-bold'>{reco.title}</span>
              <p className="mt-4 text-justify">{reco.plot.substring(0, 250)}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default App;
