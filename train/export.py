"""Exportiert tirenet.pt → tirenet.onnx (Eingabe RGB 0..1 [1,3,224,224], Ausgabe [1,5]: p,cx,cy,w,h)."""
import shutil
import numpy as np
import onnxruntime as ort
import torch
from train import TireNet, ROOT, INPUT

model = TireNet().eval()
model.load_state_dict(torch.load(ROOT / "tirenet.pt", map_location="cpu"))
x = torch.rand(1, 3, INPUT, INPUT)
path = ROOT / "tirenet.onnx"
torch.onnx.export(model, x, str(path), input_names=["images"], output_names=["out"], opset_version=17, dynamo=False)
ref = model(x).detach().numpy()
got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"images": x.numpy()})[0]
print("max. Abweichung PyTorch↔ONNX:", float(np.abs(ref - got).max()))
print("Größe:", round(path.stat().st_size / 1e6, 2), "MB")
shutil.copy(path, ROOT.parent / "web" / "models" / "tirenet.onnx")
print("nach web/models kopiert")
