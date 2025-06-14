# gaussian-occupancy-prediction-with-nuscene

environment
```python
nvidia-smi
Sat Jun 14 15:53:31 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 576.52                 Driver Version: 576.52         CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce GTX 1650      WDDM  |   00000000:03:00.0  On |                  N/A |
| 50%   38C    P8              8W /   75W |     384MiB /   4096MiB |      7%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------
```

### run result ###
```python
======
Loading NuScenes tables for version v1.0-trainval...
23 category,
8 attribute,
4 visibility,
64386 instance,
12 sensor,
10200 calibrated_sensor,
2631083 ego_pose,
68 log,
850 scene,
34149 sample,
2631083 sample_data,
1166187 sample_annotation,
4 map,
Done loading in 68.971 seconds.
======
Reverse indexing ...
Done reverse indexing in 50.5 seconds.
======
  0%|          | 3/34149 [00:01<5:01:35,  1.89it/s]
[DEBUG] Loaded 3 multiview samples
Traceback (most recent call last):
  File "...\train.py", line 132, in <module>
    occ_feat = occupancy_decoder(gaussian_embed, voxel_coords)
  File "...\.venv\lib\site-packages\torch\nn\modules\module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "...\gaussian_encoder\gaussian_decoder\gaussian_decoder.py", line 50, in forward
    dist = chunked_cdist(xyz, anchor_grid, chunk_size=512)  # (Nv, N)
  File "...\utils\ops.py", line 20, in chunked_cdist
    dist_chunk = torch.cdist(chunk, anchor_grid)  # (chunk_size, N)
  File "...\.venv\lib\site-packages\torch\functional.py", line 1222, in cdist
    return _VF.cdist(x1, x2, p, None)  # type: ignore[attr-defined]
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 48.00 MiB (GPU 0; 4.00 GiB total capacity; 6.88 GiB already allocated; 0 bytes free; 6.94 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

since rtx1650 is 4gib, it seems hard to train. But it seems that the training process is well-structured.(decoder part is quite far along in the process)


### citation ###

@article{nuscenes2019,
  title={nuScenes: A multimodal dataset for autonomous driving},
  author={Holger Caesar and Varun Bankiti and Alex H. Lang and Sourabh Vora and 
          Venice Erin Liong and Qiang Xu and Anush Krishnan and Yu Pan and 
          Giancarlo Baldan and Oscar Beijbom},
  journal={arXiv preprint arXiv:1903.11027},
  year={2019}
}

@article{fong2021panoptic,
  title={Panoptic nuScenes: A Large-Scale Benchmark for LiDAR Panoptic Segmentation and Tracking},
  author={Fong, Whye Kit and Mohan, Rohit and Hurtado, Juana Valeria and Zhou, Lubing and Caesar, Holger and
          Beijbom, Oscar and Valada, Abhinav},
  journal={arXiv preprint arXiv:2109.03805},
  year={2021}
}
