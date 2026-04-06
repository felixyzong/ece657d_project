# ece657d_project
ECE 657D Final Project

Student: Felix Zong 20873157

## testing machine
CPU: Intel 14500kf

RAM: 16GB

GPU: RTX 3070ti

## requirements
```{}
torch = 2.11.0+cu126
torchvision = 0.26.0+cu126
```

## train teacher
```{bash}
$ python ./source/teacher/train.py
```

## train student baseline
```{bash}
$ python ./source/student_baseline/train.py
```

## train student with standard distillation with varying dataset size
```{bash}
$ python ./source/student_distilled_100/train.py
$ python ./source/student_distilled_50/train.py
$ python ./source/student_distilled_20/train.py
$ python ./source/student_distilled_10/train.py
```

## train student with data free distillation
```
$ python ./source/student_distilled_00_deep_inversion/train.py
$ python ./source/student_distilled_00_generator_resnet/train_generator.py
$ python ./source/student_distilled_00_generator_resnet/train.py
```
