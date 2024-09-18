# Introduction

The code is used for demonstration purposes of KAN only.
Most of the code are modified from the original KAN document by Dr.Kang Ruiyuan.

# Structure
in totol, five examples are provided herein, and stored into five folders,respectively. In each code,a general description and necessary comments are provided inside the python code.

fit_case1: where we showed how KAN could be used for fitting the data generated from a simple function, and how to extract physical insights from KAN

mass_velocity_1: where we showed that although KAN can fit the data from Einstein's mass-velocity equation well from scractch, it may be not physically meaningful and generalizable.

mass_velocity_2: where we showed that by incooperating with domain knowledge-driven features, KAN could rediscover the Einstein's mass-velocity equation

mass_velocity_3: where we showed that if uncertainty exists between data and theory, how to embed the theory into the KAN, and finetune it to learn the uncertainty.

Beer_Lambert_law: where we compared the symolic learning performance between KAN and PySR.

# Requirements
In order to provide a simple operation condition, we tested the code on a windows machine without GPU.
We eliminated the need of other python packages, all the needed ones are listed below and shown in requirements.txt
* matplotlib==3.9.2
* numpy==2.1.1
* pykan==0.2.6
* pysr==0.19.4
* sympy==1.11.1
* torch