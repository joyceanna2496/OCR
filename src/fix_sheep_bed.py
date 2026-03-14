import sys

mapping = {
    "1": "NC_040252.1",
    "2": "NC_040253.1",
    "3": "NC_040254.1",
    "4": "NC_040255.1",
    "5": "NC_040256.1",
    "6": "NC_040257.1",
    "7": "NC_040258.1",
    "8": "NC_040259.1",
    "9": "NC_040260.1",
    "10": "NC_040261.1",
    "11": "NC_040262.1",
    "12": "NC_040263.1",
    "13": "NC_040264.1",
    "14": "NC_040265.1",
    "15": "NC_040266.1",
    "16": "NC_040267.1",
    "17": "NC_040268.1",
    "18": "NC_040269.1",
    "19": "NC_040270.1",
    "20": "NC_040271.1",
    "21": "NC_040272.1",
    "22": "NC_040273.1",
    "23": "NC_040274.1",
    "24": "NC_040275.1",
    "25": "NC_040276.1",
    "26": "NC_040277.1",
    "X": "NC_040278.1"
}

input_bed = sys.argv[1]
output_bed = sys.argv[2]

with open(input_bed) as f, open(output_bed, "w") as out:
    for line in f:
        chrom, start, end = line.strip().split()
        new_chrom = mapping.get(chrom, chrom)
        out.write(f"{new_chrom}\t{start}\t{end}\n")