import utils
import gpt_2_simple as gpt

model_name = "124M"

def finetune(corp_files, steps):

    sess = gpt.start_tf_sess()

    for c in corp_files:

        gpt.finetune(sess, "Corpora/" + c, model_name=model_name, steps=steps, print_every=int(steps/2), run_name=c.replace(".csv", ""))
