## GT (VoxelFEM)
* bridge
* cantilever flexion E

### bridge:
```
python3 training/train_voxelfem.py --grid "[320, 160, 80]" --prob "problems/3d/bridge.json" --mgl 3 --iter 5000 --optim "OC" --v0 "0.4"
```

### cantilever_flexion_e
```
python3 training/train_voxelfem.py --grid "[256, 128, 128]" --prob "problems/3d/cantilever_flexion.json" --mgl 3 --iter 5000 --optim "OC" --v0 "0.5"
```

## FF (Explicit Design Representation Network)
* bridge, s=1.0
* bridge, s=2.5
* cantilever flexion E, s=4.0

### bridge, s=1.0
```
python3 training/train_xdg.py --grid "[320, 160, 80]" --prob "problems/3d/bridge.json" --mgl 3 --iter 5000 --v0 "0.4" --es 1024 --nn 512 --nl 4 --sigma 1.0 --cs 100
```

### bridge, s=2.5
```
python3 training/train_xdg.py --grid "[320, 160, 80]" --prob "problems/3d/bridge.json" --mgl 3 --iter 5000 --v0 "0.4" --es 1024 --nn 512 --nl 4 --sigma 2.5 --cs 100
```

### cantilever flexion E, s=4.0
```
python3 training/training/train_xdg.py --grid "[256, 128, 128]" --prob "problems/3d/cantilever_flexion.json" --mgl 3 --iter 5000 --v0 "0.5" --es 1024 --nn 512 --nl 4 --sigma 4.0 --cs 100
```
