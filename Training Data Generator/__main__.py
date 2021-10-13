# have finetuning corpora in some folder entitled corpora
# autoinstall all dependencies
# work off command line
# spit out training data into output folder

# arguments will be temperature range, length of outputs
# start_temp end_temp step_size output_length


import sys
import gpt_2_simple as gpt
import os

import finetune
import generate

model_name = "124M"


if __name__ == "__main__":

    opt = sys.argv[1]  # -f -g -b finetune, generate, both (?)
    args = sys.argv[2:]

    s_temp = float(args[0])
    e_temp = float(args[1])
    num_temps = int(args[2])

    out_length = int(args[3])

    step = (e_temp - s_temp)/num_temps

    temps = [s_temp + i*step for i in range(num_temps)]

    if not os.path.isdir("Corpora"):

        raise Exception("No Corpora directory")

    extensions = [os.path.splitext(x)[1] for x in os.listdir("Corpora")]

    if set(extensions) != {'.csv'}:

        print(set(extensions))
        raise Exception("all files in Corpora directory must be .csv files")


    corp_files = [f for f in os.listdir("Corpora")]


# if gpt not downloaded, download now

if not os.path.isdir(os.path.join("models", model_name)):

    gpt.download_gpt2(model_name=model_name)

# go through temps + corpora, checking which have been done already

finetune.finetune(corp_files, 1000)
generate.generate(temps, corp_files, 50000)



