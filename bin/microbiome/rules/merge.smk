# Imports
import os
import glob

# Config params
configfile: "./config/main.yaml"
workdir: config["workdir"]
tooldir = config["tooldir"]

# Get all FASTQ files in the directory
fastq_dir = config["fastq"]["dir"]
kraken2_db = config["kraken2"]["db"]

fastq_files = []
for i in range(1, 21):
    barcode_txt = f"barcode{str(i).zfill(2)}"
    if i < 11:
        fastq_files.extend(glob.glob(f"{fastq_dir}/chili1/*/fastq_pass/{barcode_txt}/*.fastq.gz", recursive=True))
    else:
        fastq_files.extend(glob.glob(f"{fastq_dir}/chili2/*/fastq_pass/{barcode_txt}/*.fastq.gz", recursive=True))

# Writing input file names for logging purposes
# with open("input-files.txt", "w+") as input_log:
#     for name in fastq_files:
#         input_log.write(name + "\n")


# Extract sample names from the filenames
barcodes = [fastq_file.split("/")[-1].split("_")[2] for fastq_file in fastq_files]
barcodes = set(barcodes)
filenames = [fastq_file.split("/")[-1].split(".")[0] for fastq_file in fastq_files]
chilis = ["chili1", "chili2"]
dates = ["20250205_1236_MN36331_FBA39386_39421a12", "20250205_1236_MN19155_FBA42409_7f95ef4d"]

rule all:
    input:
        expand("{fastq_dir}/output/kraken2/merged/{barcode}.tsv", barcode=barcodes, fastq_dir=fastq_dir)

rule merge:
    resources:
        mem_mb=12000
    conda:
        "microbiome"
    input:
        fastq_dir=fastq_dir
    output:
        "{fastq_dir}/output/kraken2/merged/{barcode}.tsv"
    shell:
        """
        python3 /commons/dsls/spicy/bin/microbiome/scripts/kraken-tool.py \
            --path {fastq_dir}/output/kraken2/ \
            --barcode {wildcards.barcode} \
            --out {fastq_dir}/output/kraken2/merged
        """