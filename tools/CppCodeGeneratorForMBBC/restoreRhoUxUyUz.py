import re
import numpy as np

def restoreRhoUxUyUz(index, normal, fk, full):
	lines = []
	normalCode = (normal[0] + 5) * 100 + (normal[1] + 5) * 10 + (normal[2] + 5)
	if index == 0:
		lines.append("__host__ __device__ void restoreRhoUxUyUz(")
		lines.append("	const int &outerNormalX, const int &outerNormalY, const int &outerNormalZ,")
		lines.append("	float &rho, float &ux, float &uy, float &uz,")
		lines.append("	const float (&f)[27]")
		lines.append(")")
		lines.append("{")
		lines.append("	const int normalCode = (outerNormalX + 5) * 100 + (outerNormalY + 5) * 10 + (outerNormalZ + 5);")
		lines.append("	if ( normalCode == " + str(normalCode) + " )")
	else:
		lines.append("	if ( normalCode == " + str(normalCode) + " )")
	lines.append("	{")
	lines.append("		// Multiply (Sq C^T Q)^-1 Sq C^T Mfk fk")
	for i, row in enumerate(full):
		line = "		const float q" + str(i) + " ="
		for j, multiplier in enumerate(full[i]):
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
		line += ";"
		lines.append(line)
		
	lines.append("		rho = q0;")
	lines.append("		ux = q1 / rho;")
	lines.append("		uy = q2 / rho;")
	lines.append("		uz = q3 / rho;")
	lines.append("	}")
	
	return lines
