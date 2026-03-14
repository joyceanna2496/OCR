from pyfaidx import Fasta
import argparse

def bed_to_fasta(bed_file, ref_file, out_fasta):
    genome = Fasta(ref_file)

    with open(bed_file) as bed, open(out_fasta, "w") as out:
        for line in bed:
            chrom, start, end = line.strip().split()
            start = int(start)
            end = int(end)
            seq = genome[chrom][start:end]

            out.write(f">{chrom}:{start}-{end}\n{str(seq)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bed", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    bed_to_fasta(args.bed, args.ref, args.out)