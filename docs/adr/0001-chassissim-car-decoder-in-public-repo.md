# Decode ChassisSim car files in the public repo

FullModelSim reads the binary `.car` files of ChassisSim directly, with a decoder in this public repository, so that a ChassisSim car loads in one step. The owner chose this on 18 November 2026 over a private decoder that writes a plain-text car sheet, and over a private fork. The decoder makes public a reverse-engineered layout of a proprietary format, so the owner checks the ChassisSim licence terms. Vendor car files and other ChassisSim data never go into the repository. Tests use synthetic files only.

The same module also encodes, because the desktop app saves an edited setup as a new car file. It never overwrites a source car file. Encoding a decoded vendor file must give back the same bytes, before any edit is saved.

## Considered Options

- A private decoder that writes a car sheet in our own text format, and FullModelSim reads only car sheets. It keeps the layout private and also allows hand-filled sheets.
- A private fork of the whole repository with the decoder built in. It doubles the code bases to keep in step.
