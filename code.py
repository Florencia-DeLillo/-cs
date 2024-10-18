#!/usr/bin/env python

import itertools
from collections import defaultdict
from mhfp.encoder import MHFPEncoder
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdmolops import GetDistanceMatrix


def to_smiles(mol):
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

def get_atom_envs(mol, radius):
    atoms_env = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atoms_env[idx] = [find_env(mol, idx, r) for r in range(1, radius + 1)]
    return atoms_env

def find_env(mol, idx, radius):
    env = rdmolops.FindAtomEnvironmentOfRadiusN(mol, radius, idx)
    atom_map = {}

    submol = Chem.PathToSubmol(mol, env, atomMap=atom_map)
    if idx in atom_map:
        smiles = Chem.MolToSmiles(submol, rootedAtAtom=atom_map[idx], canonical=True, isomericSmiles=False)
        return smiles
    return ''

def all_pairs(mol, atoms_env, radius, is_counted=False):
    atom_pairs = []
    distance_matrix = GetDistanceMatrix(mol)
    num_atoms = mol.GetNumAtoms()
    shingle_dict = defaultdict(int)
    for idx1, idx2 in itertools.combinations(range(num_atoms), 2):
        dist = str(int(distance_matrix[idx1][idx2]))

        for i in range(radius):
            env_a = atoms_env[idx1][i]
            env_b = atoms_env[idx2][i]

            ordered = sorted([env_a, env_b])

            shingle = '{}|{}|{}'.format(ordered[0], dist, ordered[1])

            if is_counted:
                shingle_dict[shingle] += 1
                shingle += '|' + str(shingle_dict[shingle])

            atom_pairs.append(shingle.encode('utf-8'))
    return list(set(atom_pairs))

def calculate_fingerprint(smiles, dimensions=1024, radius=2, is_counted=False, is_folded=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    encoder = MHFPEncoder(dimensions)
    atom_env_pairs = all_pairs(mol, get_atom_envs(mol, radius), radius, is_counted)

    # Calculate the MinHash fingerprint using the encoder's hash function
    fingerprint = encoder.hash(set(atom_env_pairs))

    if is_folded:
        # Optionally fold the fingerprint to reduce dimensionality
        return encoder.fold(fingerprint, dimensions)

    return fingerprint


if __name__ == "__main__":
    # Example SMILES string
    smiles = 'c1ccccc1'  # Benzene

    # Calculate fingerprint for the molecule
    fingerprint = calculate_fingerprint(smiles, dimensions=1024, radius=2)

    # Print the fingerprint
    print(f"SMILES: {smiles}")
    print(f"Fingerprint: {fingerprint}")
