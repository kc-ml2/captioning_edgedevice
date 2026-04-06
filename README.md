# Captioning Edge Device

Lightweight image captioning system optimized for edge devices using BLIP / InstructBLIP and MobileVLM.  
Supports multiple inference backends including PyTorch and ONNX with quantization (INT4/INT8).

---

## 📂 Project Structure
```bash
captioning_edgedevice/ 
├── MobileVLM-v2-1.7b/     # MobileVLM (cloned from official repo)
├── blip-serving/          # BLIP / InstructBLIP server implementations 
├── mobilevlm-runtime/     # PyTorch / ONNX runtime implementations 
├── onnx_tutorial/         # ONNX conversion and usage examples 
├── README.md 
└── requirements.txt

git clone https://github.com/xxx/MobileVLM.git MobileVLM-v2-1.7b
```
