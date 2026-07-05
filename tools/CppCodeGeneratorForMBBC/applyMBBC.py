import re
import numpy as np

def applyMBBC(index, normal, fk, fu, inv, Sm, Smfk):
	lines = []
	normalCode = (normal[0] + 5) * 100 + (normal[1] + 5) * 10 + (normal[2] + 5)
	if index == 0:
		lines.append("__host__ __device__ void applyMBBC(")
		lines.append("	const int &outerNormalX, const int &outerNormalY, const int &outerNormalZ,")
		lines.append("	const float &rho, const float &ux, const float &uy, const float &uz,")
		lines.append("	float (&f)[27]")
		lines.append(")")
		lines.append("{")
		lines.append("	const int normalCode = (outerNormalX + 5) * 100 + (outerNormalY + 5) * 10 + (outerNormalZ + 5);")
		lines.append("	if ( normalCode == " + str(normalCode) + " )")
	else:
		lines.append("	if ( normalCode == " + str(normalCode) + " )")
	lines.append("	{")
	lines.append("		// Multiply S Mfk fk")
	for i, row in enumerate(Smfk):
		line = "		const float smf" + str(i) + " ="
		for j, multiplier in enumerate(row):
			if multiplier == 0:
				continue
			if multiplier == 1:
				line += " + " + fk[j]
			elif multiplier == -1:
				line += " - " + fk[j]
		if line[-1] == "=":
			line += " 0.f;"
		else:
			line += ";"
		lines.append(line)
		
	lines.append("		// Calculate equilibrium moments")
	for i, moment in enumerate(Sm):
		line = "		const float m" + str(i) + " = "
		line += moment
		line += ";"
		lines.append(line)
		
	lines.append("		// Subtract Sm - S Mfk fk")
	for i, moment in enumerate(Sm):
		line = "		const float s" + str(i) + " = "
		line += "m" + str(i) + " - smf" + str(i) + ";"
		lines.append(line)

	lines.append("		// Multiply (S Mfu)^-1 * Smf to get unknown distributions")
	for i, unknown in enumerate(fu):
		line = "		" + unknown + " ="
		for j, multiplier in enumerate(inv[i]):
			if multiplier == "0":
				continue
			elif multiplier == "1":
				line += " + s" + str(j)
				continue
			elif multiplier == "-1":
				line += " - s" + str(j)
				continue
			else:
				line += " + "
				line += multiplier
				line += " * "
				line += "s" + str(j)
		if line[-1] == "=":
			line += " 0.f"
		line += ";"
		lines.append(line)
	lines.append("	}")
	
	return lines
