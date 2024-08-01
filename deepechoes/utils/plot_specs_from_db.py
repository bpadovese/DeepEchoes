import tables
import argparse
import random
from pathlib import Path
from deepechoes.utils.image_transforms import unscale_data
from matplotlib import pyplot as plt

def plot_single_spec(hdf5_db, train_table, index=None):
    db = tables.open_file(hdf5_db, 'r')
    table = db.get_node(train_table + '/data')
    print(table.attrs.min_value)
    print(table.attrs.max_value)

    # If no index is provided, randomly sample one
    if index is None:
        index = random.randint(0, table.nrows - 1)

    mel_spectrogram = unscale_data(table[index]['data'])  

    plt.figure(figsize=(4, 3))
    plt.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("single_spectrogram.png")
    plt.show()
    db.close()

def plot_specs(hdf5_db, train_table, random_sample=True):
    db = tables.open_file(hdf5_db, 'r')
    table = db.get_node(train_table + '/data')

    # total number of records in the table
    num_records = table.nrows

    if random_sample:
        # Randomly sample indices without replacement (im using 16 but could be any number)
        sampled_indices = random.sample(range(num_records), min(16, num_records))
    else:
        # Get the first 16 entries or up to the number available if fewer than 16
        sampled_indices = list(range(min(16, num_records)))

    fig, axs = plt.subplots(4, 4, figsize=(34, 28))
    plt.subplots_adjust(wspace=0, hspace=0)  # Adjust as needed
    for i, idx in enumerate(sampled_indices):
        ax = axs[i // 4, i % 4]
        mel_spectrogram = unscale_data(table[idx]['data'])  
        ax.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='viridis')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("128.png")
    plt.show()
    plt.close(fig)
    db.close()

def main():
    
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'
    
    parser = argparse.ArgumentParser(description="Plot spectrograms from HDF5 dataset.")
    parser.add_argument("db", type=str, help="Path to the HDF5 database file")
    parser.add_argument("table", type=str, help="Name of the table within the HDF5 database")
    parser.add_argument("--random_sample", type=boolean_string, default=True, help="Randomly sample from the db or get the first 16")
    parser.add_argument("--single", type=boolean_string, default=False, help="Plot a single spectrogram")
    parser.add_argument("--index", type=int, default=None, help="Index of the spectrogram to plot (only if plotting a single one)")
    args = parser.parse_args()
    
    if args.single:
        plot_single_spec(args.db, args.table, args.index)
    else:
        plot_specs(args.db, args.table, args.random_sample)

if __name__ == "__main__":
    main()