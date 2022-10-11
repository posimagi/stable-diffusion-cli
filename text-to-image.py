#!/usr/bin/python3
# make sure you're logged in with `huggingface-cli login`
import argparse
import subprocess
import glob
import os
import shutil
import sys
import torch
from diffusers import StableDiffusionPipeline

def main():
	parser = argparse.ArgumentParser(description="Stable Diffusion text-to-image pipeline")
	parser.add_argument("-p", "--prompt", type=str, help="The prompt from which to generate an image")
	parser.add_argument("-d", "--display", action='store_true', help="Display the most recently-generated image")
	parser.add_argument("-k", "--keep", action='store_true', help="Move the most recently-generated image to keepers")
	args = parser.parse_args()

	if len(sys.argv) < 2:
		parser.print_help()
	
	if args.display:
		list_of_files = glob.glob('output/*.png')
		latest_file = os.path.basename(max(list_of_files, key=os.path.getctime))
		subprocess.run(["explorer.exe", latest_file], cwd="output")

	if args.keep:
		list_of_files = glob.glob('output/*.png')
		latest_file = max(list_of_files, key=os.path.getctime)
		shutil.move(latest_file, "keepers/")

	if args.prompt:
		prompt = args.prompt

		# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_type=torch.float16, revision="fp16")
		pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
		pipe = pipe.to("cuda")
		pipe.enable_attention_slicing()

		filename = os.path.join("output", '-'.join(prompt.split(' ')) + ".png")
		image = pipe(prompt, height=504, width=896).images[0]  # maximum safe 16:9 size
		image.save(filename)

if __name__ == "__main__":
	main()