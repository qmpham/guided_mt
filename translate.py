import numpy as np
import os
import argparse
import ctranslate2
import yaml

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', required=False, help="config")
    argparser.add_argument('-s', '--src', required=False, help="Source file path")
    argparser.add_argument('-t', '--tgt', required=False, help="Target file path")
    argparser.add_argument('-m', '--model', required=True, help="Model path")
    argparser.add_argument('-j', '--joiner', default='‖', help="Sentence joiner")
    argparser.add_argument('-d', '--dev', default='cpu', choices=["cpu", "cuda", "auto"], help="Device to translate")
    argparser.add_argument('--hyp', required=False, help="Output translation file name")
    args = argparser.parse_args()
    with open(args.config,"r") as stream:
        config = yaml.load(stream)

    for src_file, tgt_file in zip(config["src"], config["tgt"]):
        print("translate %s with prefix %s"%(src_file,tgt_file))
        with open(src_file,"r") as f:
            src = [l.strip().split() for l in f.readlines()]
        with open(tgt_file,"r") as f:
            tgt_all = [l.strip() for l in f.readlines()]
        prefix = []
        for i in range(len(tgt_all)):
            if args.joiner in tgt_all[i]:
                pre = " ".join([tgt_all[i], args.joiner])
                prefix.append(pre.split())
            else:
                prefix.append(None)


        translator = ctranslate2.Translator(
                model_path=args.model,
                device=args.dev,
                device_index=0)

        output = translator.translate_batch(
                source=src,
                target_prefix=prefix,
                max_batch_size=60,
                beam_size=5,
                length_penalty=0.2,
                min_decoding_length=0,
                return_scores=False)
        import os
        save_path = os.path.dirname(args.model) 
        src_name = os.path.basename(src_file)
        hyp_file = os.path.join(save_path,"eval","%s.trans"%src_name)
        print("output: %s"%hyp_file)
        with open(hyp_file, "w") as f:
            for batch in output:
                for line in batch:
                    hyp = " ".join(line["tokens"])
                    # f.write(" ".join(line["tokens"]) + "\n")
                    f.write(hyp.split(" %s " % args.joiner)[-1] + "\n")
