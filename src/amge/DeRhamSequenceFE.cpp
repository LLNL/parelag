/*
  Copyright (c) 2015, Lawrence Livermore National Security, LLC. Produced at the
  Lawrence Livermore National Laboratory. LLNL-CODE-669695. All Rights reserved.
  See file COPYRIGHT for details.

  This file is part of the ParElag library. For more information and source code
  availability see http://github.com/LLNL/parelag.

  ParElag is free software; you can redistribute it and/or modify it under the
  terms of the GNU Lesser General Public License (as published by the Free
  Software Foundation) version 2.1 dated February 1999.
*/

#include <numeric>
#include "elag_amge.hpp"

class VolumetricFEMassIntegrator;
class PointFEMassIntegrator;

DeRhamSequenceFE::DeRhamSequenceFE(AgglomeratedTopology * topo, int nforms):
		DeRhamSequence(topo, nforms),
		owns_data(false),
		di(nforms-1),
		mi(nforms, nforms),
		fecColl(nforms),
		feSpace(nforms),
		mesh(NULL)
{

}

void DeRhamSequenceFE::Build(ParMesh * mesh_, Array<FiniteElementSpace *> & feSpace_, Array<DiscreteInterpolator *> & di_, Array2D<BilinearFormIntegrator *> & mi_)
{
	di.MakeRef(di_);
	mi.MakeRef(mi_);
	feSpace.MakeRef(feSpace_);
	mesh = mesh_;
	owns_data = false;

	buildDof();
	assembleLocalMass();
	assembleDerivative();
}

void  DeRhamSequenceFE::ReplaceMassIntegrator(AgglomeratedTopology::EntityByCodim ientity, int jform, BilinearFormIntegrator * m, bool recompute)
{
	elag_assert(jform >= jformStart);

	//TODO This method is NOT very smart :(
    delete mi(ientity, jform);
	mi(ientity, jform) = m;

	if(recompute)
	{
		for(int icodim(0); icodim < mi.NumRows(); ++icodim)
			for(int jform(0); jform < mi.NumCols()-icodim; ++jform)
			{
				delete M(icodim,jform);
				M(icodim,jform) = static_cast<SparseMatrix *>(NULL);
			}

		assembleLocalMass();
	}
}

SparseMatrix * DeRhamSequenceFE::ComputeProjectorFromH1ConformingSpace(int jform)
{
	FiniteElementSpace * conformingH1space;

	if(jform == nforms-1)
		conformingH1space = feSpace[0];
	else
		conformingH1space = new FiniteElementSpace(mesh, feSpace[0]->FEColl(), mesh->Dimension(), Ordering::byVDIM);

	SparseMatrix * pi;
	DiscreteLinearOperator * id = new DiscreteLinearOperator(conformingH1space, feSpace[jform]);
	id->AddDomainInterpolator(new IdentityInterpolator() );
	id->Assemble();
	id->Finalize();
	pi = id->LoseMat();

	if(jform != nforms-1)
		delete conformingH1space;

	delete id;
	return pi;
}

DeRhamSequenceFE::~DeRhamSequenceFE()
{
	if(owns_data)
	{
		for(int i(0); i < di.Size(); ++i)
			delete di[i];

		for(int icodim(0); icodim < mi.NumRows(); ++icodim)
			for(int jform(0); jform < mi.NumCols()-icodim; ++jform)
				delete mi(icodim,jform);

		for(int i(0); i < feSpace.Size(); ++i)
			delete feSpace[i];

		for(int i(0); i < fecColl.Size(); ++i)
			delete fecColl[i];
	}
}

void DeRhamSequenceFE::buildDof()
{
	int codim_base;
	for(int jform(jformStart); jform < nforms; ++jform)
	{
		codim_base = nforms - jform -1;
//		std::cout<<"Build FE dof for space " << feSpace[jform]->FEColl()->Name() << "(codim_base = "<< codim_base << ")" << std::endl;
		dof[jform] = new DofHandlerFE(mesh->GetComm(), feSpace[jform], codim_base);
		dof[jform]->BuildEntityDofTables();
	}
}

void DeRhamSequenceFE::assembleLocalMass()
{
//#ifdef ELEMAGG_USE_OPENMP
//	assembleLocalMass_omp();
//#else
	assembleLocalMass_ser();
//#endif
}

void DeRhamSequenceFE::assembleLocalMass_old()
{

	ElementTransformation *eltrans;
	const FiniteElement * fe;
	DenseMatrix Mloc;
	int * I, *J, *J_it;
	double *A, * A_it;
	int nnz, nLocDof;

	for(int icodim(0); icodim < topo->Codimensions()+1; ++icodim)
		for(int jform(jformStart); jform < nforms - icodim; ++jform)
		{
//			std::cout << " Assemble mass matrix on entity_type " << icodim << " for space " << feSpace[jform]->FEColl()->Name() << std::endl;
			const SparseMatrix & entity_dof = dof[jform]->GetEntityDofTable( static_cast<AgglomeratedTopology::EntityByCodim>(icodim) );
			const int nEntities  = entity_dof.Size();
			const int nRdof      = entity_dof.NumNonZeroElems();

			I = new int[nRdof+1];
			nnz = 0;

			I[0] = nnz;
			Array<int> rdofs;
			for(int ientity(0); ientity < nEntities; ++ientity)
			{
				dof[jform]->GetrDof(static_cast<AgglomeratedTopology::EntityByCodim>(icodim), ientity, rdofs);
				nLocDof = rdofs.Size();
				nnz += nLocDof*nLocDof;
				for(int * idof(rdofs.GetData()); idof != rdofs.GetData()+nLocDof; ++idof)
					I[*idof+1] = nLocDof;
			}

			std::partial_sum(I, I+nRdof+1, I);

			J = new int[nnz];
			A = new double[nnz];

			for(int ientity(0); ientity < nEntities; ++ientity)
			{
				fe = GetFE(jform, icodim, ientity);
				eltrans = GetTransformation(icodim, ientity);
				nLocDof = fe->GetDof();
				dof[jform]->GetrDof(static_cast<AgglomeratedTopology::EntityByCodim>(icodim), ientity, rdofs);
				mi(icodim, jform)->AssembleElementMatrix(*fe, *eltrans, Mloc);
				for(int iloc(0); iloc < nLocDof; ++iloc)
				{
					int idof = rdofs[iloc];
					J_it = J+I[ idof ];
					A_it = A+I[ idof ];
					for(int jloc(0); jloc < nLocDof; ++jloc)
					{
						*(J_it++) = rdofs[jloc];
						*(A_it++) = Mloc(iloc, jloc);
					}
				}
			}

			M(icodim, jform) = new SparseMatrix(I,J,A,nRdof, nRdof);

		}

#if 0
	{
	std::ofstream fid("M.dat");
	for(int icodim(0); icodim < nforms; ++icodim)
		for(int jform(0); jform < nforms-icodim; ++jform)
		{
			fid << "Mass matrix on entity_type " << icodim << " for space " << feSpace[jform]->FEColl()->Name() << std::endl;
			M(icodim, jform)->PrintMatlab(fid);
		}
	}
#endif
}

void DeRhamSequenceFE::assembleLocalMass_ser()
{
	ElementTransformation *eltrans;
	const FiniteElement * fe;
	DenseMatrix * Mloc;
	int nLocDof;

	for(int icodim(0); icodim < topo->Codimensions()+1; ++icodim)
		for(int jform(jformStart); jform < nforms - icodim; ++jform)
		{
			const SparseMatrix & entity_dof = dof[jform]->GetEntityDofTable( static_cast<AgglomeratedTopology::EntityByCodim>(icodim) );
			const int nEntities  = entity_dof.Size();
            ElementalMatricesContainer mass(nEntities);
			for(int ientity(0); ientity < nEntities; ++ientity)
			{
				fe = GetFE(jform, icodim, ientity);
				eltrans = GetTransformation(icodim, ientity);
				nLocDof = fe->GetDof();
                Mloc = new DenseMatrix(nLocDof);
				mi(icodim, jform)->AssembleElementMatrix(*fe, *eltrans, *Mloc);
                mass.SetElementalMatrix(ientity, Mloc);
			}
			M(icodim, jform) = mass.GetAsSparseMatrix();
		}

#if 0
	{
	std::ofstream fid("M.dat");
	for(int icodim(0); icodim < nforms; ++icodim)
		for(int jform(0); jform < nforms-icodim; ++jform)
		{
			fid << "Mass matrix on entity_type " << icodim << " for space " << feSpace[jform]->FEColl()->Name() << std::endl;
			M(icodim, jform)->PrintMatlab(fid);
		}
	}
#endif
}


void DeRhamSequenceFE::assembleLocalMass_omp()
{
	std::cout << "assembleLocalMass_opm\n";
	int * I, *J;
	double *A;
	int nnz, nLocDof;

	for(int icodim(0); icodim < topo->Codimensions()+1; ++icodim)
		for(int jform(jformStart); jform < nforms - icodim; ++jform)
		{
//			std::cout << " Assemble mass matrix on entity_type " << icodim << " for space " << feSpace[jform]->FEColl()->Name() << std::endl;
			const SparseMatrix & entity_dof = dof[jform]->GetEntityDofTable( static_cast<AgglomeratedTopology::EntityByCodim>(icodim) );
			const int nEntities  = entity_dof.Size();
			const int nRdof      = entity_dof.NumNonZeroElems();

			I = new int[nRdof+1];
			nnz = 0;

			I[0] = nnz;
			{
				Array<int> rdofs;
				for(int ientity(0); ientity < nEntities; ++ientity)
				{
					dof[jform]->GetrDof(static_cast<AgglomeratedTopology::EntityByCodim>(icodim), ientity, rdofs);
					nLocDof = rdofs.Size();
					nnz += nLocDof*nLocDof;
					for(int * idof(rdofs.GetData()); idof != rdofs.GetData()+nLocDof; ++idof)
						I[*idof+1] = nLocDof;
				}
			}

			std::partial_sum(I, I+nRdof+1, I);

			J = new int[nnz];
			A = new double[nnz];

#pragma omp parallel default(shared) private(nLocDof)
			{
				IsoparametricTransformation *eltrans = new IsoparametricTransformation;
				const FiniteElement * fe;
				DenseMatrix Mloc;
				Array<int> rdofs;

				int *J_it;
				double *A_it;

				int idof, iloc, jloc;

#pragma omp for schedule(guided)
				for(int ientity = 0; ientity < nEntities; ++ientity)
				{
					fe = GetFE(jform, icodim, ientity);
					GetTransformation(icodim, ientity, *eltrans);
					nLocDof = fe->GetDof();
					dof[jform]->GetrDof(static_cast<AgglomeratedTopology::EntityByCodim>(icodim), ientity, rdofs);
					mi(icodim, jform)->AssembleElementMatrix(*fe, *eltrans, Mloc);
					for(iloc = 0; iloc < nLocDof; ++iloc)
					{
						idof = rdofs[iloc];
						J_it = J+I[ idof ];
						A_it = A+I[ idof ];
						for(jloc = 0; jloc < nLocDof; ++jloc)
						{
							*(J_it++) = rdofs[jloc];
							*(A_it++) = Mloc(iloc, jloc);
						}
					}
				}
			}

			M(icodim, jform) = new SparseMatrix(I,J,A,nRdof, nRdof);

		}
}


void DeRhamSequenceFE::assembleDerivative()
{
	for(int jform(jformStart); jform < nforms-1; ++jform)
	{
//		std::cout << " Assemble Differential Operator from "<< feSpace[jform]->FEColl()->Name() << " to " << feSpace[jform+1]->FEColl()->Name() << std::endl;
		DiscreteLinearOperator interp(feSpace[jform], feSpace[jform+1]);
		interp.AddDomainInterpolator(di[jform]);
		interp.Assemble();
		interp.Finalize();
		LoseInterpolators(interp);
		D[jform] = interp.LoseMat();
	}

#if 0
	{
	std::ofstream fid("D.dat");
	for(int jform(0); jform < nforms-1; ++jform)
	{
		fid << "Differential Operator from "<< feSpace[jform]->FEColl()->Name() << " to " << feSpace[jform]->FEColl()->Name() << std::endl;
		D[jform]->PrintMatlab(fid);
	}
	}
#endif
}

void DeRhamSequenceFE::showP(int jform, SparseMatrix & P, Array<int> & parts)
{
	elag_assert(jform < nforms);
	elag_assert(jform >= jformStart );
	elag_assert(feSpace[jform]->GetNDofs() == P.Size())

	int num_procs, myid;
	MPI_Comm comm = mesh->GetComm();

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myid);


		   GridFunction u(feSpace[jform]);
		   Array<int> cols;
		   Vector srow;
		   SparseMatrix * P_t = Transpose(P);
		   char vishost[] = "localhost";
		   int  visport   = 19916;
		   osockstream sock (visport, vishost);
		   for(int i(0); i < P.Width(); ++i)
		   {
			   u = 0.0;
			   P_t->GetRow(i, cols, srow);
			   u.SetSubVector(cols, srow);
			   sock << "parallel " << num_procs << " " << myid << "\n";
			   sock << "solution\n";
			   this->mesh->PrintWithPartitioning(parts.GetData(), sock);
			   u.Save(sock);
			   sock.send();
			   sock << "pause \n";
		   }

		   delete P_t;
}

void DeRhamSequenceFE::show(int jform, MultiVector & v)
{
	elag_assert(jform < nforms);
	elag_assert(jform >= jformStart );
	elag_assert(feSpace[jform]->GetNDofs() == v.Size())

	int num_procs, myid;
	MPI_Comm comm = mesh->GetComm();

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myid);

	GridFunction u;
	Vector v_view;
	char vishost[] = "localhost";
	int  visport   = 19916;
	osockstream sock (visport, vishost);

	for(int i(0); i < v.NumberOfVectors(); ++i)
	{
	   v.GetVectorView(i, v_view);
	   u.Update(feSpace[jform], v_view, 0);
	   sock << "parallel " << num_procs << " " << myid << "\n";
	   sock << "solution\n";
	   this->mesh->Print(sock);
	   u.Save(sock);
	   sock.send();
	   sock << "pause \n";
	}
}

MultiVector * DeRhamSequenceFE::InterpolateScalarTargets(int jform, const Array<Coefficient *> & targets)
{
	elag_assert( jform == 0 || jform == nforms - 1) // jform is a VectorFiniteElement!

	int size = feSpace[jform]->GetNDofs();
	int nTargets = targets.Size();

	MultiVector * t = new MultiVector(nTargets, size);
	Vector tview;
	GridFunction gf;

	for(int itarget(0); itarget < nTargets; ++itarget)
	{
		t->GetVectorView(itarget, tview);
		gf.Update(feSpace[jform], tview, 0);
		gf.ProjectCoefficient(*targets[itarget]);
	}

	return t;
}

MultiVector * DeRhamSequenceFE::InterpolateVectorTargets(int jform, const Array<VectorCoefficient *> & targets)
{
	elag_assert( jform != 0 && jform != nforms - 1); // jform is a ScalarFiniteElement!

	int size = feSpace[jform]->GetNDofs();
	int nTargets = targets.Size();

	MultiVector * t = new MultiVector(nTargets, size);
	Vector tview;
	GridFunction gf;

	for(int itarget(0); itarget < nTargets; ++itarget)
	{
		t->GetVectorView(itarget, tview);
		gf.Update(feSpace[jform], tview, 0);
		gf.ProjectCoefficient(*targets[itarget]);
	}

	return t;
}

void DeRhamSequenceFE::ProjectCoefficient(int jform, Coefficient & c, Vector & v)
{
	v.SetSize(feSpace[jform]->GetNDofs());
	GridFunction gf;
	gf.Update(feSpace[jform], v, 0);
	gf.ProjectCoefficient(c);
}

void DeRhamSequenceFE::ProjectVectorCoefficient(int jform, VectorCoefficient & c, Vector & v)
{
	v.SetSize(feSpace[jform]->GetNDofs());
	GridFunction gf;
	gf.Update(feSpace[jform], v, 0);
	gf.ProjectCoefficient(c);
}

const FiniteElement * DeRhamSequenceFE::GetFE(int jform, int ientity_type, int ientity) const
{
	if(jform + ientity_type > nforms)
		mfem_error("Wrong combination of jform and ientity_type");

	if(mesh->Dimension() == 3)
	{
		switch(ientity_type)
		{
		case AgglomeratedTopology::ELEMENT:
			return feSpace[jform]->GetFE(ientity);
			break;
		case AgglomeratedTopology::FACET:
			return feSpace[jform]->FEColl()->FiniteElementForGeometry(mesh->GetFaceBaseGeometry(ientity));
			break;
		case AgglomeratedTopology::RIDGE:
			return feSpace[jform]->FEColl()->FiniteElementForGeometry(Geometry::SEGMENT);
		case AgglomeratedTopology::PEAK:
			return feSpace[jform]->FEColl()->FiniteElementForGeometry(Geometry::POINT);
			break;
		default:
			mfem_error("Wrong ientity_type");
			return NULL;
		}
	}
	else
	{
		switch(ientity_type)
		{
		case AgglomeratedTopology::ELEMENT:
			return feSpace[jform]->GetFE(ientity);
			break;
		case AgglomeratedTopology::FACET:
			return feSpace[jform]->FEColl()->FiniteElementForGeometry(Geometry::SEGMENT);
		case AgglomeratedTopology::RIDGE:
			return feSpace[jform]->FEColl()->FiniteElementForGeometry(Geometry::POINT);
			break;
		default:
			mfem_error("Wrong ientity_type");
			return NULL;
		}
	}
}

ElementTransformation * DeRhamSequenceFE::GetTransformation(int ientity_type, int ientity) const
{
	if(mesh->Dimension() == 3)
	{
		switch(ientity_type)
		{
		case AgglomeratedTopology::ELEMENT:
			return mesh->GetElementTransformation(ientity);
			break;
		case AgglomeratedTopology::FACET:
			return mesh->GetFaceTransformation(ientity);
			break;
		case AgglomeratedTopology::RIDGE:
			return mesh->GetEdgeTransformation(ientity);
			break;
		case AgglomeratedTopology::PEAK:
			return NULL;
			break;
		default:
			mfem_error("Wrong ientity_type");
			return NULL;
		}
	}
	else
	{
		switch(ientity_type)
		{
		case AgglomeratedTopology::ELEMENT:
			return mesh->GetElementTransformation(ientity);
			break;
		case AgglomeratedTopology::FACET:
			return mesh->GetEdgeTransformation(ientity);
		case AgglomeratedTopology::RIDGE:
			return NULL;
			break;
		default:
			mfem_error("Wrong ientity_type");
			return NULL;
		}
	}
}

void DeRhamSequenceFE::GetTransformation(int ientity_type, int ientity, IsoparametricTransformation & tr) const
{
	if(mesh->Dimension() == 3)
	{
		switch(ientity_type)
		{
		case AgglomeratedTopology::ELEMENT:
			mesh->GetElementTransformation(ientity, &tr);
			break;
		case AgglomeratedTopology::FACET:
			mesh->GetFaceTransformation(ientity, &tr);
			break;
		case AgglomeratedTopology::RIDGE:
			mesh->GetEdgeTransformation(ientity, &tr);
			break;
		case AgglomeratedTopology::PEAK:
			tr.GetPointMat().SetSize(mesh->Dimension(), 0);
			tr.SetFE(NULL);
			break;
		default:
			mfem_error("Wrong ientity_type");
		}
	}
	else
	{
		switch(ientity_type)
		{
		case AgglomeratedTopology::ELEMENT:
			mesh->GetElementTransformation(ientity, &tr);
			break;
		case AgglomeratedTopology::FACET:
			mesh->GetEdgeTransformation(ientity, &tr);
			break;
		case AgglomeratedTopology::RIDGE:
			tr.GetPointMat().SetSize(mesh->Dimension(), 0);
			tr.SetFE(NULL);
			break;
		default:
			mfem_error("Wrong ientity_type");
		}
	}
}

//-----------------------------------------------------------------------------

DeRhamSequence3D_FE::DeRhamSequence3D_FE(AgglomeratedTopology * topo, ParMesh * mesh_, int order):
		DeRhamSequenceFE(topo, 4)
{

	mesh = mesh_;
	int nDimensions = mesh->Dimension();

	if(nDimensions != 3)
		mfem_error("We are not in 3D!\n");

	fecColl[0] = new H1_FECollection(order+1, nDimensions );
	fecColl[1] = new ND_FECollection(order+1, nDimensions );
	fecColl[2] = new RT_FECollection(order, nDimensions );
	fecColl[3] = new L2_FECollection(order, nDimensions );

	for(int i(0); i < nDimensions+1; ++ i)
		this->feSpace[i] = new FiniteElementSpace(mesh, fecColl[i]);

	this->di[0] = new GradientInterpolator();
	this->di[1] = new CurlInterpolator();
	this->di[2] = new DivergenceInterpolator2();

	//Integrators on volumes
	mi(0,0) = new MassIntegrator();
	mi(0,1) = new VectorFEMassIntegrator();
	mi(0,2) = new VectorFEMassIntegrator();
	mi(0,3) = new MassIntegrator();

	//Integrators on faces
	mi(1,0) = new MassIntegrator();
	mi(1,1) = new ND_3D_FacetMassIntegrator();
	mi(1,2) = new VolumetricFEMassIntegrator();

	//Integrators on edges
	mi(2,0) = new MassIntegrator();
	mi(2,1) = new VolumetricFEMassIntegrator();

	//Integrators on points
	mi(3,0) = new PointFEMassIntegrator();

	owns_data = true;

	buildDof();
	assembleLocalMass();
	assembleDerivative();

}

DeRhamSequence3D_FE::~DeRhamSequence3D_FE()
{

}

void DeRhamSequence3D_FE::computePVTraces(AgglomeratedTopology::EntityByCodim icodim, Vector & pv)
{
	int jform = nforms - 1 - icodim;
	switch(icodim)
	{
	case AgglomeratedTopology::ELEMENT:
		InterpolatePV_L2(feSpace[jform], topo->AEntityEntity(AgglomeratedTopology::ELEMENT),pv);
		break;
	case AgglomeratedTopology::FACET:
		InterpolatePV_HdivTraces(feSpace[jform], topo->AEntityEntity(AgglomeratedTopology::FACET),pv);
		break;
	case AgglomeratedTopology::RIDGE:
		InterpolatePV_HcurlTraces(feSpace[jform], topo->AEntityEntity(AgglomeratedTopology::RIDGE),pv);
		break;
	case AgglomeratedTopology::PEAK:
		InterpolatePV_H1Traces(feSpace[jform], topo->AEntityEntity(AgglomeratedTopology::PEAK),pv);
		break;
	default:
		mfem_error("DeRhamSequence3D_FE::computePVTraces(Topology::entity icodim, Vector & pv)");
	}
}

DeRhamSequence2D_Hdiv_FE::DeRhamSequence2D_Hdiv_FE(AgglomeratedTopology * topo, ParMesh * mesh_, int order):
		DeRhamSequenceFE(topo, 3)
{
mesh = mesh_;
	int nDimensions = mesh->Dimension();

	if(nDimensions != 2)
		mfem_error("We are not in 2D!\n");

	fecColl[0] = new H1_FECollection(order+1, nDimensions );
	fecColl[1] = new RT_FECollection(order, nDimensions );
	fecColl[2] = new L2_FECollection(order, nDimensions );


	for(int i(0); i < nDimensions+1; ++ i)
		this->feSpace[i] = new FiniteElementSpace(mesh, fecColl[i]);

	this->di[0] = new GradientInterpolator();
	this->di[1] = new DivergenceInterpolator2();

	//Integrators on faces
	mi(0,0) = new MassIntegrator();
	mi(0,1) = new VectorFEMassIntegrator();
	mi(0,2) = new MassIntegrator();

	//Integrators on edges
	mi(1,0) = new MassIntegrator();
	mi(1,1) = new VolumetricFEMassIntegrator();


	//Integrators on points
	mi(2,0) = new PointFEMassIntegrator();

	owns_data = true;

	buildDof();
	assembleLocalMass();
	assembleDerivative();

}

void DeRhamSequence2D_Hdiv_FE::computePVTraces(AgglomeratedTopology::EntityByCodim icodim, Vector & pv)
{
	int jform = nforms - 1 - icodim;
	switch(icodim)
	{
	case AgglomeratedTopology::ELEMENT:
		InterpolatePV_L2(feSpace[jform], topo->AEntityEntity(AgglomeratedTopology::ELEMENT),pv);
		break;
	case AgglomeratedTopology::FACET:
		InterpolatePV_HdivTraces(feSpace[jform], topo->AEntityEntity(AgglomeratedTopology::FACET),pv);
		break;
	case AgglomeratedTopology::RIDGE:
		InterpolatePV_H1Traces(feSpace[jform], topo->AEntityEntity(AgglomeratedTopology::RIDGE),pv);
		break;
	default:
		mfem_error("DeRhamSequence2D_Hdiv_FE::computePVTraces(Topology::entity icodim, Vector & pv)");
	}
}

DeRhamSequence2D_Hdiv_FE::~DeRhamSequence2D_Hdiv_FE()
{

}

void InterpolatePV_L2(const FiniteElementSpace * fespace, const SparseMatrix & AE_element, Vector & AE_Interpolant)
{
	GridFunction gf;
	gf.Update(const_cast<FiniteElementSpace *>(fespace), AE_Interpolant, 0);
	ConstantCoefficient ones(1);

	gf.ProjectCoefficient(ones);
}


void InterpolatePV_HdivTraces(const FiniteElementSpace * fespace, const SparseMatrix & AF_facet, Vector & AF_Interpolant)
{

	const int nAF = AF_facet.Size();
	const int ndofs = fespace->GetNDofs();
	Mesh * mesh = fespace->GetMesh();
	int nDim = mesh->Dimension();

	if(AF_Interpolant.Size() != ndofs)
		mfem_error("InterpolatePV_HdivTraces(const FiniteElementSpace * fespace, const SparseMatrix & AF_facet, Vector & AF_Interpolant)");

	const int    * i_AF_facet = AF_facet.GetI();
	const int    * j_AF_facet = AF_facet.GetJ();
	const double * a_AF_facet = AF_facet.GetData();

	Array<int> vdofs(0);

	const FiniteElement * fe;
	const IntegrationRule * nodes;

	Vector localVector;
	for(int iAF(0); iAF < nAF; ++iAF)
	{
		for(const int * ifc = j_AF_facet + i_AF_facet[iAF]; ifc != j_AF_facet + i_AF_facet[iAF+1]; ++ifc, ++a_AF_facet)
		{
			 fe = fespace->FEColl()->FiniteElementForGeometry(mesh->GetFaceBaseGeometry(*ifc));
			 int ndofs = fe->GetDof();
			 vdofs.SetSize(ndofs);

			 if(nDim == 3)
				 fespace->GetFaceVDofs(*ifc, vdofs);
			 else /* nDim == 2 */
				 fespace->GetEdgeVDofs(*ifc, vdofs);

	         ElementTransformation * tr = mesh->GetFaceTransformation(*ifc);
	         nodes = &fe->GetNodes();
	         localVector.SetSize(ndofs);
	         for (int i = 0; i < ndofs; i++)
	         {
	            const IntegrationPoint &ip = nodes->IntPoint(i);
	            tr->SetIntPoint(&ip);
	            localVector(i) = *(a_AF_facet)*tr->Weight();
	         }
	         AF_Interpolant.SetSubVector(vdofs,localVector);
		}
	}


}

void InterpolatePV_HcurlTraces(const FiniteElementSpace * fespace, const SparseMatrix & AR_ridge, Vector & AR_Interpolant)
{
	const int nAR = AR_ridge.Size();
	const int ndofs = fespace->GetNDofs();
	Mesh * mesh = fespace->GetMesh();

	if(AR_Interpolant.Size() != ndofs)
		mfem_error("InterpolatePV_HcurlTraces(const FiniteElementSpace * fespace, const SparseMatrix & AF_facet, Vector & AR_Interpolant)");

	const int    * i_AR_ridge = AR_ridge.GetI();
	const int    * j_AR_ridge = AR_ridge.GetJ();
	const double * a_AR_ridge = AR_ridge.GetData();

	const FiniteElement * fe = fespace->FEColl()->FiniteElementForGeometry(Geometry::SEGMENT);
	const IntegrationRule * nodes = &fe->GetNodes();

	const int ndof_ridge = fe->GetDof();

	AR_Interpolant.SetSize( ndofs );
	AR_Interpolant = 0.0;


	Array<int> vdofs(ndof_ridge);
	Vector localVector;

	for(int iAR(0); iAR < nAR; ++iAR)
	{
		for(const int * irg = j_AR_ridge + i_AR_ridge[iAR]; irg != j_AR_ridge + i_AR_ridge[iAR+1]; ++irg, ++a_AR_ridge)
		{
	         fespace->GetEdgeVDofs(*irg, vdofs);
	         ElementTransformation * tr = mesh->GetEdgeTransformation(*irg);
	         localVector.SetSize(ndof_ridge);
	         for (int i = 0; i < ndof_ridge; i++)
	         {
	            const IntegrationPoint &ip = nodes->IntPoint(i);
	            tr->SetIntPoint(&ip);
	            localVector(i) = *(a_AR_ridge)*tr->Weight();
	         }
	         AR_Interpolant.SetSubVector(vdofs, localVector);
		}
	}

}

void InterpolatePV_H1Traces(const FiniteElementSpace * fespace, const SparseMatrix & AP_peak, Vector & AP_Interpolant)
{
	int ndofs = fespace->GetNDofs();
	int nnz = AP_peak.NumNonZeroElems();

	if(AP_Interpolant.Size() != ndofs)
		mfem_error("InterpolatePV_H1Traces(const FiniteElementSpace * fespace, const SparseMatrix & AF_facet, Vector & AP_Interpolant)");

	AP_Interpolant.SetSize(ndofs);
	AP_Interpolant = 0.0;

	int * j_AP_peak = AP_peak.GetJ();
	int * end = j_AP_peak+nnz;

	double * val = AP_Interpolant.GetData();
	for(; j_AP_peak != end; ++j_AP_peak)
		val[*j_AP_peak] =1.0;
}

void DeRhamSequenceFE::DEBUG_CheckNewLocalMassAssembly()
{
	MPI_Comm comm;
	std::stringstream os;

	Array2D<SparseMatrix *> M_ser( M.NumRows(), M.NumCols() );
	Array2D<SparseMatrix *> M_old( M.NumRows(), M.NumCols() );
	M_old = static_cast<SparseMatrix *>(NULL);
	Swap(M, M_old);

	StopWatch chrono;

	chrono.Start();
	assembleLocalMass_ser();
	chrono.Stop();
	os << "assembleLocalMass_ser took " << chrono.RealTime() << "\n";
	Swap(M, M_ser);

	chrono.Clear();
	chrono.Start();
	assembleLocalMass_old();
	chrono.Stop();
	os << "assembleLocalMass_old took " << chrono.RealTime() << "\n";
	Swap(M, M_old);

	for(int icodim(0); icodim < nforms; ++icodim)
		for(int jform(0); jform < nforms - icodim; ++jform)
		{
			std::stringstream name_ser;
			std::stringstream name_new;
			name_ser << "M_ser(" << icodim <<" , " << jform << ")";
			name_ser << "M_old(" << icodim <<" , " << jform << ")";
			AreAlmostEqual( *(M_ser(icodim,jform)), *(M_old(icodim,jform)), name_ser.str(), name_new.str(), 1e-9, true, os);

			delete M_ser(icodim,jform);
			delete M_old(icodim, jform);

		}


	comm = topo->GetComm();
	SerializedOutput(comm, std::cout, os.str());

}
