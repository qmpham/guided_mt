import numpy as np
import os
import argparse
import ctranslate2

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-s', '--src', required=True, help="Source file path")
    argparser.add_argument('-t', '--tgt', required=True, help="Target file path")
    argparser.add_argument('-m', '--model', required=True, help="Model path")
    argparser.add_argument('-j', '--joiner', default='※', help="Sentence joiner")
    argparser.add_argument('-d', '--dev', default='cpu', choices=["cpu", "cuda", "auto"], help="Device to translate")
    argparser.add_argument('--hyp', required=True, help="Output translation file name")
    args = argparser.parse_args()
    src = [l.split() for l in open(args.src)]
    tgt_all = [l for l in open(args.tgt)]
    prefix = []
    for i in range(len(tgt_all)):
        if args.joiner in tgt_all[i]:
            pre = tgt_all[i].split(args.joiner)[0] + args.joiner
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

    with open(args.hyp, "w") as f:
        for batch in output:
            for line in batch:
                hyp = " ".join(line["tokens"])
                # f.write(" ".join(line["tokens"]) + "\n")
                f.write(hyp.split(" %s " % args.joiner)[-1] + "\n")