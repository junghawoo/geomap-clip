<div align="center">    
 
# 🌎 GeoMapCLIP: Worldwide Map Image Geo-localization
GeoMapCLIP is fine-tuned GeoCLIP. 

![ALT TEXT](/figures/GeoMapCLIP.png)

</div>


## Description
This project aims to develop a vision system capable of interpreting map images and extracting bounding box coordinates to describe map content.
By fine-tuning models such as CLIP and applying image retrieval techniques, the system will learn to recognize legends, scales, symbols, and coordinate grids within geospatial maps, enabling automated extraction of structured spatial information.
This advancement will significantly enhance the geospatial reasoning capabilities of large language models, supporting researchers in efficiently querying and leveraging legacy geospatial datasets.
Initial development will focus on fine-tuning GeoCLIP (Contrastive Language-Image Pretraining) for map-specific tasks. The vision system will produce a list of coordinates corresponding to visual elements in map figures.

![ALT TEXT](/figures/method.png)

## Method

Similarly to OpenAI's CLIP, GeoMapCLIP is trained contrastively by matching Image-GPS pairs. 
GeoMapCLIP learns distinctive visual features associated with different locations on earth.

_🚧 Repo Under Construction 🔨_

## 📎 Getting Started: API

You can install GeoMapCLIP's module using pip:

```
pip install geomapclip
```

or directly from source:

```
git clone https://github.com/junghawoo/geomap-clip
cd geomap-clip
python setup.py install
```

## 🗺️📍 Worldwide Image Geolocalization

![ALT TEXT](/figures/inference.png)

### Usage: GeoMapCLIP Inference

```python
import torch
from geomapclip import GeoMapCLIP

model = GeoMapCLIP()

image_path = "image.png"

top_pred_gps, top_pred_prob = model.predict(image_path, top_k=5)

print("Top 5 GPS Predictions")
print("=====================")
for i in range(5):
    lat, lon = top_pred_gps[i]
    print(f"Prediction {i+1}: ({lat:.6f}, {lon:.6f})")
    print(f"Probability: {top_pred_prob[i]:.6f}")
    print("")
```


![ALT TEXT](/figures/downstream-task.png)

### Usage: Pre-Trained Location Encoder

```python
import torch
from geomapclip import LocationEncoder

gps_encoder = LocationEncoder()

gps_data = torch.Tensor([[40.7128, -74.0060], [34.0522, -118.2437]])  # NYC and LA in lat, lon
gps_embeddings = gps_encoder(gps_data)
print(gps_embeddings.shape) # (2, 512)
```

