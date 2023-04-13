import csv

# Lee el archivo csv y crea un diccionario con los IDs como claves y las secuencias como valores
csv_file = '/home/oem/Desktop/ALEX/Project/data/csv_output.csv'
sequence_dict = {}
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Salta la primera fila de encabezado
    for row in csv_reader:
        sequence_dict[row[0]] = row[1]

# Lee el archivo fasta y crea un diccionario con los IDs como claves y las secuencias como valores
fasta_file = '/home/oem/Desktop/ALEX/Project/data/multifasta.fa'
fasta_dict = {}
with open(fasta_file, 'r') as file:
    current_id = ''
    current_sequence = ''
    for line in file:
        if line.startswith('>'):
            if current_id != '':
                fasta_dict[current_id] = current_sequence
            current_id = line.strip()[1:].split()[0]
            current_sequence = ''
        else:
            current_sequence += line.strip()
    fasta_dict[current_id] = current_sequence

# Crea un nuevo archivo csv con una columna adicional que contiene la secuencia fasta correspondiente al ID
output_file = '/home/oem/Desktop/ALEX/Project/data/all_data.csv'
with open(output_file, 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['ID', '1d_seq', 'fasta_seq'])
    for key, value in sequence_dict.items():
        fasta_sequence = fasta_dict[key]
        csv_writer.writerow([key, value, fasta_sequence])

