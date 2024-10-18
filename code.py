#!/usr/bin/env python

import itertools
from collections import defaultdict
import tmap as tm
from mhfp.encoder import MHFPEncoder
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdmolops import GetDistanceMatrix

def to_smiles(mol):
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

class MAP4Calculator:

    def __init__(self, dimensions=1024, radius=2, is_counted=False, is_folded=False, return_strings=False):
        self.dimensions = dimensions
        self.radius = radius
        self.is_counted = is_counted
        self.is_folded = is_folded
        self.return_strings = return_strings

        if self.is_folded:
            self.encoder = MHFPEncoder(dimensions)
        else:
            self.encoder = tm.Minhash(dimensions)

    def calculate(self, mol):
        atom_env_pairs = self._calculate(mol)
        if self.is_folded:
            return self._fold(atom_env_pairs)
        elif self.return_strings:
            return atom_env_pairs
        return self.encoder.from_string_array(atom_env_pairs)

    def calculate_many(self, mols):
        atom_env_pairs_list = [self._calculate(mol) for mol in mols]
        if self.is_folded:
            return [self._fold(pairs) for pairs in atom_env_pairs_list]
        elif self.return_strings:
            return atom_env_pairs_list
        return self.encoder.batch_from_string_array(atom_env_pairs_list)

    def _calculate(self, mol):
        return self._all_pairs(mol, self._get_atom_envs(mol))

    def _fold(self, pairs):
        fp_hash = self.encoder.hash(set(pairs))
        return self.encoder.fold(fp_hash, self.dimensions)

    def _get_atom_envs(self, mol):
        atoms_env = {}
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            for radius in range(1, self.radius + 1):
                if idx not in atoms_env:
                    atoms_env[idx] = []
                atoms_env[idx].append(MAP4Calculator._find_env(mol, idx, radius))
        return atoms_env

    @classmethod
    def _find_env(cls, mol, idx, radius):
        env = rdmolops.FindAtomEnvironmentOfRadiusN(mol, radius, idx)
        atom_map = {}

        submol = Chem.PathToSubmol(mol, env, atomMap=atom_map)
        if idx in atom_map:
            smiles = Chem.MolToSmiles(submol, rootedAtAtom=atom_map[idx], canonical=True, isomericSmiles=False)
            return smiles
        return ''

    def _all_pairs(self, mol, atoms_env):
        atom_pairs = []
        distance_matrix = GetDistanceMatrix(mol)
        num_atoms = mol.GetNumAtoms()
        shingle_dict = defaultdict(int)
        for idx1, idx2 in itertools.combinations(range(num_atoms), 2):
            dist = str(int(distance_matrix[idx1][idx2]))

            for i in range(self.radius):
                env_a = atoms_env[idx1][i]
                env_b = atoms_env[idx2][i]

                ordered = sorted([env_a, env_b])

                shingle = '{}|{}|{}'.format(ordered[0], dist, ordered[1])

                if self.is_counted:
                    shingle_dict[shingle] += 1
                    shingle += '|' + str(shingle_dict[shingle])

                atom_pairs.append(shingle.encode('utf-8'))
        return list(set(atom_pairs))


if __name__ == "__main__":
    # Initialize MAP4Calculator with default parameters
    calculator = MAP4Calculator(dimensions=1024, radius=2, is_counted=False, is_folded=False)

    # Define your SMILES strings here
    smiles_list = ['c1ccccc1', 'c1cccc(N)c1']

    # Convert SMILES to RDKit mol objects
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    # Calculate fingerprints for all molecules
    fingerprints = calculator.calculate_many(mols)

    # Print the fingerprints
    for i, fp in enumerate(fingerprints):
        print(f"SMILES: {smiles_list[i]}")
        print(f"Fingerprint: {fp}")
