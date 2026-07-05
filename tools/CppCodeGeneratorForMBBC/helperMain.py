import numpy as np
import copy
import sympy as sp
import os
import re

from applyMBBC import *
from restoreRhoUxUyUz import *
from restoreUxUyUz import *
from restoreRho import *
from applyMirror import *
from getLatex import *

f    =  [ f"f_{{{i}}}" for i in range(27) ]
cx   =  [ 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 ]
cy   =  [ 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 ]
cz   =  [ 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 ]

w    =  [ "8/27", 
		  "2/27", "2/27", "2/27 ", "2/27", "2/27", "2/27", 
		  "1/54", "1/54", "1/54", "1/54", "1/54", "1/54", "1/54", "1/54", "1/54", "1/54", "1/54", "1/54", 
		  "1/216", "1/216", "1/216", "1/216", "1/216", "1/216", "1/216", "1/216" ]

abcs =	[ "000", 
		  "100", "010", "001", 
		  "200", "020", "002", "011", "101", "110", 
		  "111", "210", "201", "120", "021", "102", "012", 
		  "022", "202", "220", "211", "121", "112" ]

meq = [ "rho",
		"rho * ux",
		"rho * uy",
		"rho * uz",
		"(1.f/3.f) * rho + rho * ux * ux",
		"(1.f/3.f) * rho + rho * uy * uy",
		"(1.f/3.f) * rho + rho * uz * uz",
		"rho * uy * uz",
		"rho * ux * uz",
		"rho * ux * uy",
		"0.f",
		"(1.f/3.f) * rho * uy + rho * ux * ux * uy",
		"(1.f/3.f) * rho * uz + rho * ux * ux * uz",
		"(1.f/3.f) * rho * ux + rho * ux * uy * uy",
		"(1.f/3.f) * rho * uz + rho * uy * uy * uz",
		"(1.f/3.f) * rho * ux + rho * ux * uz * uz",
		"(1.f/3.f) * rho * uy + rho * uy * uz * uz",
		"(1.f/9.f) * rho + (1.f/3.f) * rho * uy * uy + (1.f/3.f) * rho * uz * uz + rho * uy * uy * uz * uz",
		"(1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uz * uz + rho * ux * ux * uz * uz",
		"(1.f/9.f) * rho + (1.f/3.f) * rho * ux * ux + (1.f/3.f) * rho * uy * uy + rho * ux * ux * uy * uy",
		"(1.f/3.f) * rho * uy * uz",
		"(1.f/3.f) * rho * ux * uz",
		"(1.f/3.f) * rho * ux * uy"
		]

meqOrder = [0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]

Q = sp.Matrix([
	[1, 0, 0, 0],
	[0, 1, 0, 0],
	[0, 0, 1, 0],
	[0, 0, 0, 1],
	[sp.Rational(1, 3), 0, 0, 0],
	[sp.Rational(1, 3), 0, 0, 0],
	[sp.Rational(1, 3), 0, 0, 0],
	[0, 0, 0, 0],
	[0, 0, 0, 0],
	[0, 0, 0, 0],
	[0, 0, 0, 0],
	[0, 0, sp.Rational(1, 3), 0],
	[0, 0, 0, sp.Rational(1, 3)],
	[0, sp.Rational(1, 3), 0, 0],
	[0, 0, 0, sp.Rational(1, 3)],
	[0, sp.Rational(1, 3), 0, 0],
	[0, 0, sp.Rational(1, 3), 0],
	[sp.Rational(1, 9), 0, 0, 0],
	[sp.Rational(1, 9), 0, 0, 0],
	[sp.Rational(1, 9), 0, 0, 0],
	[0, 0, 0, 0],
	[0, 0, 0, 0],
	[0, 0, 0, 0],
])

Qu = sp.Matrix([
	[0, 0, 0],
	[1, 0, 0],
	[0, 1, 0],
	[0, 0, 1],
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
	[0, sp.Rational(1, 3), 0],
	[0, 0, sp.Rational(1, 3)],
	[sp.Rational(1, 3), 0, 0],
	[0, 0, sp.Rational(1, 3)],
	[sp.Rational(1, 3), 0, 0],
	[0, sp.Rational(1, 3), 0],
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
	[0, 0, 0],
])

Qrho = sp.Matrix([
	[1],
	[0],
	[0],
	[0],
	[sp.Rational(1, 3)],
	[sp.Rational(1, 3)],
	[sp.Rational(1, 3)],
	[0],
	[0],
	[0],
	[0],
	[0],
	[0],
	[0],
	[0],
	[0],
	[0],
	[sp.Rational(1, 9)],
	[sp.Rational(1, 9)],
	[sp.Rational(1, 9)],
	[0],
	[0],
	[0],
])

def sympyToCppStrings(M, bracketsEverywhere = False):
	result = []
	for row in M.tolist():
		new_row = []
		for x in row:
			s = str(x)
			if "/" in s:
				num, den = s.split("/")
				s = f"({num}.f/{den}.f)"
			elif s not in ["0", "1", "-1"] or bracketsEverywhere:
				# make integers explicit floats if desired
				s = f"({s}.f)"
			new_row.append(s)
		result.append(new_row)
	return result

def getBCdata(normal):
	fk = []
	fu = []

	cxyz = np.vstack((np.array(cx), np.array(cy), np.array(cz))).T
	identifier = np.min(cxyz * normal, axis=1)
	fkIndexes = np.where(identifier >= 0)[0]
	fuIndexes = np.where(identifier < 0)[0]
	for i in fkIndexes:
		fk.append(f[i])
	for i in fuIndexes:
		fu.append(f[i])

	mfk = []
	mfu = []

	def getMoment(mfk, mfu, abc):
		a, b, c = int(abc[0]), int(abc[1]), int(abc[2])
		cResult = np.array(cx, dtype=int)**a * np.asarray(cy, dtype=int)**b * np.asarray(cz, dtype=int)**c
		mfkRow = (cResult[fkIndexes]).tolist()
		mfk.append(mfkRow)
		mfuRow = (cResult[fuIndexes]).tolist()
		mfu.append(mfuRow)

	for abc in abcs:
		getMoment(mfk, mfu, abc)
		
	mfk = sp.Matrix(mfk)
	mfu = sp.Matrix(mfu)

	### First part: MBBC
	rowsToKeep = [0]
	i = 1
	while len(rowsToKeep) < len(fu):
		if i >= len(mfu):
			print("getChosenMoments Error: Ran out of independent groups")
			return None
		previousRank = mfu.row(rowsToKeep).rank()
		attemptedRank = mfu.row(rowsToKeep + [i]).rank()
		if attemptedRank > previousRank:
			rowsToKeep.append(i)
		i += 1
	identityMatrix = sp.eye(23)
	S =  identityMatrix.row(rowsToKeep)
	
	### Second part: restoring rho, ux, uy, uz
	nullspace = mfu.T.nullspace()
	def nonzero_order_sum(v):
		key = 0
		for i, val in enumerate(v):
			if val != 0:
				key += meqOrder[i]
		return key
	sorted_nullspace = sorted(nullspace, key=nonzero_order_sum)
	C = sp.Matrix.hstack(*sorted_nullspace)
	rowsToKeep = [0]
	i = 1
	while len(rowsToKeep) < 4:
		if i >= len(C.T * Q):
			print("nullspace Error: Ran out of independent columns")
			return None
		previousRank = (C.T * Q).row(rowsToKeep).rank()
		attemptedRank =  (C.T * Q).row(rowsToKeep + [i]).rank()
		if attemptedRank > previousRank:
			rowsToKeep.append(i)
		i += 1
	identityMatrix = sp.eye(len((C.T * Q).tolist()))
	Sq =  identityMatrix.row(rowsToKeep)
	
	### Third part: restoring ux, uy, uz, known rho
	rowsToKeep = [0]
	i = 1
	while len(rowsToKeep) < 3:
		if i >= len(C.T * Qu):
			print("nullspace Error: Ran out of independent columns")
			return None
		previousRank = (C.T * Qu).row(rowsToKeep).rank()
		attemptedRank =  (C.T * Qu).row(rowsToKeep + [i]).rank()
		if attemptedRank > previousRank:
			rowsToKeep.append(i)
		i += 1
	identityMatrix = sp.eye(len((C.T * Q).tolist()))
	Su =  identityMatrix.row(rowsToKeep)
	
	### Fourth part: restoring rho, known ux, uy, uz
	rowsToKeep = []
	i = 0
	while len(rowsToKeep) < 1:
		if i >= len(C.T * Qrho):
			print("nullspace Error: Ran out of independent columns")
			return None
		attemptedResult = (C.T * Qrho).row(0)
		if attemptedResult != 0:
			rowsToKeep.append(i)
		i += 1
	identityMatrix = sp.eye(len((C.T * Q).tolist()))
	Srho =  identityMatrix.row(rowsToKeep)
	
	return fk, fu, mfk, mfu, C, S, Sq, Su, Srho

os.makedirs("results", exist_ok=True)

allNormals = [	[1,0,0], [0,1,0], [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1],
				[0,1,1], [1,0,1], [1,1,0], [0,-1,-1], [-1,0,-1], [-1,-1,0],
				[0,1,-1], [1,0,-1], [1,-1,0], [0,-1,1], [-1,0,1], [-1,1,0],
				[1,1,1], [-1,-1,-1], 
				[-1,1,1], [1,-1,1], [1,1,-1], [1,-1,-1], [-1,1,-1], [-1,-1,1] ]

#### applyMBBC
allLines = []
for index, normal in enumerate(allNormals):
	fk, fu, mfk, mfu, C, S, Sq, Su, Srho = getBCdata(normal)
	for i, fi in enumerate(fk):
		match = re.search(r"\d+", fi)
		number = int(match.group())
		fk[i] = "f[" + str(number) + "]"
	for i, fi in enumerate(fu):
		match = re.search(r"\d+", fi)
		number = int(match.group())
		fu[i] = "f[" + str(number) + "]"
		
	inv = (S * mfu).inv()
	
	inv = sympyToCppStrings(inv)
	
	Sm = []
	for row in S.tolist():
		Sm.append(meq[row.index(1)])
	
	Smfk = (S * mfk).tolist()
	
	allLines += applyMBBC(index, normal, fk, fu, inv, Sm, Smfk)
allLines.append("}")
with open("results/applyMBBC.h", "w") as file:
	file.write("\n".join(allLines))
	
#### restoreRhoU
allLines = []
for index, normal in enumerate(allNormals):
	fk, fu, mfk, mfu, C, S, Sq, Su, Srho = getBCdata(normal)
	for i, fi in enumerate(fk):
		match = re.search(r"\d+", fi)
		number = int(match.group())
		fk[i] = "f[" + str(number) + "]"
		
	inv = (Sq * C.T * Q).inv()
	full = inv * Sq * C.T * mfk	
	
	full = sympyToCppStrings(full)
	
	allLines += restoreRhoUxUyUz(index, normal, fk, full)
allLines.append("}")
with open("results/restoreRhoUxUyUz.h", "w") as file:
	file.write("\n".join(allLines))
	
#### restoreU
allLines = []
for index, normal in enumerate(allNormals):
	fk, fu, mfk, mfu, C, S, Sq, Su, Srho = getBCdata(normal)
	for i, fi in enumerate(fk):
		match = re.search(r"\d+", fi)
		number = int(match.group())
		fk[i] = "f[" + str(number) + "]"
		
	inv = (Su * C.T * Qu).inv()
	inv = sympyToCppStrings(inv)
	
	SCMfk = Su * C.T * mfk
	SCMfk = sympyToCppStrings(SCMfk)
	
	SCQ = Su * C.T * Qrho
	SCQ = sympyToCppStrings(SCQ, bracketsEverywhere = True)
	
	allLines += restoreUxUyUz(index, normal, fk, inv, SCMfk, SCQ)
allLines.append("}")
with open("results/restoreUxUyUz.h", "w") as file:
	file.write("\n".join(allLines))
	
#### restoreRho
allLines = []
for index, normal in enumerate(allNormals):
	fk, fu, mfk, mfu, C, S, Sq, Su, Srho = getBCdata(normal)
	for i, fi in enumerate(fk):
		match = re.search(r"\d+", fi)
		number = int(match.group())
		fk[i] = "f[" + str(number) + "]"
	
	SCQrho = Srho * C.T * Qrho
	SCQrho = sympyToCppStrings(SCQrho)
	
	SCMfk = Srho * C.T * mfk
	SCMfk = sympyToCppStrings(SCMfk)
	
	SCQu = Srho * C.T * Qu
	
	SCQu = sympyToCppStrings(SCQu, bracketsEverywhere = True)
	
	allLines += restoreRho(index, normal, fk, SCQrho, SCMfk, SCQu)
allLines.append("}")
with open("results/restoreRho.h", "w") as file:
	file.write("\n".join(allLines))

"""
#### Latex Tables
normal = [1, -1, 1]
fk, fu, mfk, mfu, C, S, Sq, Su, Srho = getBCdata(normal)
latexCode = getLatex(f, normal, fk, fu, mfk, mfu, C, S, Sq, Q, Su, Qu, Srho, Qrho)
"""

#### applyMirror
allLines = []
for index, normal in enumerate(allNormals):
	fk, fu, mfk, mfu, C, S, Sq, Su, Srho = getBCdata(normal)
	fuIndexList = []
	for i, fi in enumerate(fu):
		match = re.search(r"\d+", fi)
		number = int(match.group())
		fu[i] = "f[" + str(number) + "]"
		fuIndexList.append(number)
	
	allLines += applyMirror(index, normal, fu, fuIndexList, cx, cy, cz)
allLines.append("}")
with open("results/applyMirror.h", "w") as file:
	file.write("\n".join(allLines))
