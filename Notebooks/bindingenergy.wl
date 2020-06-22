(* ::Package:: *)

(* ::Title:: *)
(*binding energy from EOB*)


(* ::Text:: *)
(*assumptions*)


$Assumptions = {m1 > 0, m2 > 0, r > 0, x > 0};


(* ::Text:: *)
(*define masses*)


M = m1 + m2;
\[Mu] = m1 m2 / M;
\[Nu] = \[Mu] / M;


(* ::Text:: *)
(*effective Hamiltonian, 1PN accurate*)
(*PN book-keeping parameter \[Epsilon]*)


A = 1 - \[Epsilon] 2 M / r;
He = Sqrt[ A (\[Mu]^2 + A \[Epsilon] pr^2 + \[Epsilon] L^2 / r^2) ];


(* ::Text:: *)
(*real Hamiltonian from the energy map*)


H = Sqrt[M^2(1+2\[Nu](He/\[Mu]-1))];


(* ::Text:: *)
(*PN expand H to 1PN order*)


H1PN = Series[H, {\[Epsilon], 0, 2}] // Simplify // Normal


(* ::Text:: *)
(*require circular orbits, solve for L*)


pr = 0;
D[H1PN, r] == 0

Lsol = Series[
	L /. Solve[%, L][[2]],
	{\[Epsilon], 0, 2}] // Simplify // Normal


(* ::Text:: *)
(*get x=(M \[Omega])^(2/3), solve for r(x)*)


x == (Series[
	(M D[H1PN, L] / \[Epsilon])^(2/3) /. L -> Lsol,
	{\[Epsilon], 0, 1}] // Simplify // Normal)

rsol = Series[
	r /. Solve[%, r][[2]],
	{\[Epsilon], 0, 1}] // Simplify // Normal


(* ::Text:: *)
(*cauculate EOB binding energy e=(H-M)/\[Mu]*)


eEOB = Series[
	(H - M) / \[Mu] /. L -> Lsol /. r -> rsol,
	{\[Epsilon], 0, 2}] // Simplify // Normal


(* ::Text:: *)
(*compare to PN binding energy*)


ePN = -1/2 x (1 - 1/12 (9 + \[Nu]) x);
ePN - eEOB /. \[Epsilon] -> 1 // Simplify
