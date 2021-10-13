
import utils
import gpt_2_simple as gpt
import pandas as pd

def generate(temps, corp_files, out_length):

    for f in corp_files:

        sess = gpt.start_tf_sess()

        model_name = f.replace(".csv", "")
        gpt.load_gpt2(sess, run_name=model_name)

        for t in temps:

            if not utils.output_exists(t, f):

                sample = gpt.generate(sess, run_name = model_name, nsamples=out_length, batch_size=20, prefix="<|startoftext|>",
                                      truncate="<|endoftext|>", temperature = t, return_as_list=True)

                sample = pd.DataFrame(sample)

                out_name = utils.output_file_name(t, f)

                sample.to_csv("Output/" + out_name, index=False)