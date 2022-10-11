#!/usr/bin/python3
# make sure you're logged in with `huggingface-cli login`
import argparse
import re
import subprocess
import glob
import os
import shutil
import sys
import torch
from diffusers import StableDiffusionPipeline


def newest_output() -> str:
	flist = glob.glob('output/*.png')
	f = max(flist, key=os.path.getctime)
	return f

def newest_keeper() -> str:
	flist = glob.glob('keepers/*.png')
	f = max(flist, key=os.path.getctime)
	return f

def main():
	parser = argparse.ArgumentParser(description="Stable Diffusion text-to-image pipeline")
	parser.add_argument("-p", "--prompt", type=str, help="The prompt from which to generate an image")
	parser.add_argument("-d", "--display", action='store_true', help="Display the most recently-generated image")
	parser.add_argument("-k", "--keep", action='store_true', help="Move the most recently-generated image to keepers")
	parser.add_argument("-r", "--retrieve", action='store_true', help="Display the most recently-generated keeper")
	parser.add_argument("-c", "--copy", type=str, help="Copy the most recently-generated keeper to the specified location")
	args = parser.parse_args()

	if len(sys.argv) < 2:
		parser.print_help()
	
	if args.display:
		f = newest_output()
		fname = os.path.basename(f)
		subprocess.run(["explorer.exe", fname], cwd="output")

	if args.keep:
		f = newest_output()
		shutil.move(f, "keepers/")
	
	if args.retrieve:
		f = newest_keeper()
		fname = os.path.basename(f)
		subprocess.run(["explorer.exe", fname], cwd="keepers")

	if args.copy:
		f = newest_keeper()
		shutil.copy(f, args.copy)

	if args.prompt:
		prompt = args.prompt

		# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_type=torch.float16, revision="fp16")
		pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
		pipe = pipe.to("cuda")
		pipe.enable_attention_slicing()

		filename = os.path.join("output", '-'.join(re.sub(r'[^a-zA-Z0-9 ]', '', prompt).split(' ')) + ".png")
		image = pipe(prompt, height=504, width=896).images[0]  # maximum safe 16:9 size
		image.save(filename)

if __name__ == "__main__":
	main()