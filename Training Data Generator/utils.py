def save_data():

    pass

def output_exists(temp, corp_f):

    return os.path.isfile("Output/" + output_file_name(temp, corp_f))

def output_file_name(temp, corp_f):

    # generate output file name given temp and corp file name

    temp = str(temp).replace(".", "_")

    out_name = corp_f.replace(".csv", "").append(temp).append(".csv")

    return out_name