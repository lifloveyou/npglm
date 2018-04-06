with open('dblp.run') as inp, open('dblp.run.txt', 'w') as out:
    copy = True
    for line in inp:
        if line.startswith('Epoch'):
            copy = False
        if line.startswith('Autoencoder'):
            copy = True

        if copy:
            out.write(line)
