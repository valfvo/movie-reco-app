import React, { useCallback, useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';


const ImageUpload = ({ onStateChange }) => {
    const [image, setImage] = useState(null);

    useEffect(() => {
        onStateChange(image);
    }, [image, onStateChange]);

    const onDrop = useCallback(images => {
        const file = images[0];

        const reader = new FileReader();
        reader.onload = () => {
            setImage(reader.result);
        };
        reader.readAsDataURL(file);
    }, []);

    const { getRootProps, getInputProps } = useDropzone({onDrop, accept: 'image/*', multiple: false});

    return (
    <div className='flex flex-col border border-white rounded-md px-2.5 py-2.5 h-[32rem] w-96'>
        {image && <img src={image} alt='poster' className='grow max-h-[28rem]' />}
        <div {...getRootProps({ className: 'dropzone grow rounded-md border-2 border-dashed border-teal-600 place-content-center cursor-pointer' })}>
            <input {...getInputProps()} />
            <p>Upload {image ? 'another' : 'a'} poster</p>
        </div>
    </div>
    );
  };

export default ImageUpload;
