import argparse

# 染色體名稱 mapping
mapping = {
    "1": "NC_037328.1", "2": "NC_037329.1", "3": "NC_037330.1",
    "4": "NC_037331.1", "5": "NC_037332.1", "6": "NC_037333.1",
    "7": "NC_037334.1", "8": "NC_037335.1", "9": "NC_037336.1",
    "10": "NC_037337.1", "11": "NC_037338.1", "12": "NC_037339.1",
    "13": "NC_037340.1", "14": "NC_037341.1", "15": "NC_037342.1",
    "16": "NC_037343.1", "17": "NC_037344.1", "18": "NC_037345.1",
    "19": "NC_037346.1", "20": "NC_037347.1", "21": "NC_037348.1",
    "22": "NC_037349.1", "23": "NC_037350.1", "24": "NC_037351.1",
    "25": "NC_037352.1", "26": "NC_037353.1", "27": "NC_037354.1",
    "28": "NC_037355.1", "29": "NC_037356.1",
    "X": "NC_037357.1", "MT": "NC_037358.1"
}

def main():
    parser = argparse.ArgumentParser(description="Fix BED chromosome names")
    parser.add_argument("--bed", required=True, help="Input BED file")
    parser.add_argument("--out", required=True, help="Output fixed BED file")
    args = parser.parse_args()

    with open(args.bed) as f, open(args.out, "w") as out:
        for line in f:
            chrom, start, end = line.strip().split()
            chrom = mapping.get(chrom, chrom)
            out.write(f"{chrom}\t{start}\t{end}\n")

if __name__ == "__main__":
    main()