import re
import numpy as np

def restoreUxUyUz(index, normal, fk, inv, SCMfk, SCQ):
	lines = []
	normalCode = (normal[0] + 5) * 100 + (normal[1] + 5) * 10 + (normal[2] + 5)
	if index == 0:
		lines.append("__host__ __device__ void restoreUxUyUz(")
		lines.append("	const int &outerNormalX, const int &outerNormalY, const int &outerNormalZ,")
		lines.append("	const float &rho, float &ux, float &uy, float &uz,")
		lines.append("	const float (&f)[27]")
		lines.append(")")
		lines.append("{")
		lines.append("	const int normalCode = (outerNormalX + 5) * 100 + (outerNormalY + 5) * 10 + (outerNormalZ + 5);")
		lines.append("	if ( normalCode == " + str(normalCode) + " )")
	else:
		lines.append("	if ( normalCode == " + str(normalCode) + " )")
	lines.append("	{")
	lines.append("		// Multiply SCMfk fk")
	for i, row in enumerate(SCMfk):
		line = "		const float scmf" + str(i) + " = ( "
		for j, multiplier in enumerate(SCMfk[i]):
			if multiplier == "0":
				continue
			elif multiplier == "1":
				line += " + " + fk[j]
				continue
			elif multiplier == "-1":
				line += " - " + fk[j]
				continue
			else:
				line += " + "
				line += multiplier
				line += " * "
				line += fk[j]
		if line[-1] == "=":
			line += " 0.f"
		line += ") / rho;"
		lines.append(line)
		
	lines.append("		// Subtract 1/rho scmf - scq")
	for i in range(3):
		line = "		const float s" + str(i) + " = "
		line += "scmf" + str(i) + " - " + SCQ[i][0] + ";"
		lines.append(line)

	lines.append("		// Multiply (Su C^T Qu)^-1 * s to get u")
	directions = ["x", "y", "z"]
	for i, row in enumerate(inv):
		line = "		u" + directions[i] + " ="
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
				line += " * s"
				line += str(j)
		if line[-1] == "=":
			line += " 0.f"
		line += ";"
		lines.append(line)
		
	lines.append("	}")
	
	return lines
