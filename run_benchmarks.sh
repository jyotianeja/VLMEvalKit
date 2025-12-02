#!/bin/bash

set -x

python run.py --data AI2D_TEST --model Bunny-Vision --verbose --api-nproc 64 --judge gpt-4o-impact --reuse
python run.py --data BLINK --model Bunny-Vision --verbose --api-nproc 64 --judge gpt-4o-impact --reuse
python run.py --data ChartMuseum_test --model Bunny-Vision --verbose --api-nproc 64 --judge gpt-4o-impact --reuse
python run.py --data ChartQA_TEST --model Bunny-Vision --verbose --api-nproc 64 --judge gpt-4o-impact
python run.py --data CharXiv_descriptive_val --model Bunny-Vision --verbose --api-nproc 64 --judge gpt-4o-impact --reuse
python run.py --data DocVQA_VAL --model Bunny-Vision --verbose --api-nproc 64 --judge gpt-4o-impact --reuse
python run.py --data HallusionBench --model Bunny-Vision --verbose --api-nproc 64 --judge gpt-4o-impact --reuse
python run.py --data LogicVista --model Bunny-Vision --verbose --api-nproc 64 --judge gpt-4o-impact --reuse
python run.py --data MathVerse_MINI --model Bunny-Vision --verbose --api-nproc 64 --judge gpt-4o-impact --reuse
python run.py --data MathVision_MINI --model Bunny-Vision --verbose --api-nproc 64 --judge gpt-4o-impact --reuse
python run.py --data MathVista_MINI --model Bunny-Vision --verbose --api-nproc 64 --judge gpt-4o-impact --reuse
python run.py --data MMMU_DEV_VAL --model Bunny-Vision --verbose --api-nproc 64 --judge gpt-4o-impact --reuse
python run.py --data MMStar --model Bunny-Vision --verbose --api-nproc 64 --judge gpt-4o-impact --reuse
python run.py --data MUIRBench --model Bunny-Vision --verbose --api-nproc 64 --judge gpt-4o-impact --reuse
python run.py --data OCRBench --model Bunny-Vision --verbose --api-nproc 64 --judge gpt-4o-impact --reuse
python run.py --data OlympiadBench --model Bunny-Vision --verbose --api-nproc 64 --judge gpt-4o-impact --reuse
python run.py --data ScreenSpo --model Bunny-Vision --verbose --api-nproc 64 --judge gpt-4o-impact --reuse
python run.py --data ScreenSpot_Pro --model Bunny-Vision --verbose --api-nproc 64 --judge gpt-4o-impact --reuse
python run.py --data ScreenSpot_v2 --model Bunny-Vision --verbose --api-nproc 64 --judge gpt-4o-impact --reuse
python run.py --data WeMath --model Bunny-Vision --verbose --api-nproc 64 --judge gpt-4o-impact --reuse
python run.py --data WildVision --model Bunny-Vision --verbose --api-nproc 64 --judge gpt-4o-impact --reuse
python run.py --data ZEROBench --model Bunny-Vision --verbose --api-nproc 64 --judge gpt-4o-impact --reuse

#### run Qwen-VL benchmarks
## Qwen3-VL-8B-Instruct

# python run.py --data AI2D_TEST --model Qwen3-VL-8B-Instruct --verbose --api-nproc 64
# python run.py --data BLINK --model Qwen3-VL-8B-Instruct --verbose --api-nproc 64
# python run.py --data ChartMuseum_test --model Qwen3-VL-8B-Instruct --verbose --api-nproc 64
# python run.py --data ChartQA_TEST --model Qwen3-VL-8B-Instruct --verbose --api-nproc 64
# python run.py --data CharXiv_descriptive_val --model Qwen3-VL-8B-Instruct --verbose --api-nproc 64
# python run.py --data DocVQA_VAL --model Qwen3-VL-8B-Instruct --verbose --api-nproc 64
# python run.py --data HallusionBench --model Qwen3-VL-8B-Instruct --verbose --api-nproc 64
# python run.py --data LogicVista --model Qwen3-VL-8B-Instruct --verbose --api-nproc 64
# python run.py --data MathVerse_MINI --model Qwen3-VL-8B-Instruct --verbose --api-nproc 64
# python run.py --data MathVision_MINI --model Qwen3-VL-8B-Instruct --verbose --api-nproc 64
# python run.py --data MathVista_MINI --model Qwen3-VL-8B-Instruct --verbose --api-nproc 64
# python run.py --data MMMU_DEV_VAL --model Qwen3-VL-8B-Instruct --verbose --api-nproc 64
# python run.py --data MMStar --model Qwen3-VL-8B-Instruct --verbose --api-nproc 64
# python run.py --data MUIRBench --model Qwen3-VL-8B-Instruct --verbose --api-nproc 64
# python run.py --data OCRBench --model Qwen3-VL-8B-Instruct --verbose --api-nproc 64
# python run.py --data OlympiadBench --model Qwen3-VL-8B-Instruct --verbose --api-nproc 64
# python run.py --data ScreenSpo --model Qwen3-VL-8B-Instruct --verbose --api-nproc 64
# python run.py --data ScreenSpot_Pro --model Qwen3-VL-8B-Instruct --verbose --api-nproc 64
# python run.py --data ScreenSpot_v2 --model Qwen3-VL-8B-Instruct --verbose --api-nproc 64
# python run.py --data WeMath --model Qwen3-VL-8B-Instruct --verbose --api-nproc 64
# python run.py --data WildVision --model Qwen3-VL-8B-Instruct --verbose --api-nproc 64
# python run.py --data ZEROBench --model Qwen3-VL-8B-Instruct --verbose --api-nproc 64
