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

#ifndef AGGLOMERATEDTOPOLOGYCHECK_HPP_
#define AGGLOMERATEDTOPOLOGYCHECK_HPP_

class AgglomeratedTopologyCheck {
public:

	static void ShowBadAgglomeratedEntities(int codim, AgglomeratedTopology & topo, std::ostream & os);

private:

	static void computeBettiNumbersAgglomeratedEntities(int codim, AgglomeratedTopology & topo, DenseMatrix & bettiNumbers);
#if 0
	static void computeBettiNumbersAgglomeratedElements(AgglomeratedTopology & topo, DenseMatrix & bettiNumbers);
	static void computeBettiNumbersOfAgglomeratedFacets(AgglomeratedTopology & topo, DenseMatrix & bettiNumbers);
#endif
	static void showBadAgglomeratedElements(DenseMatrix & bettiNumbers, AgglomeratedTopology & topo, std::ostream & os);
	static void showBadAgglomeratedFacets(DenseMatrix & bettiNumbers, AgglomeratedTopology & topo, std::ostream & os);
	static void showBadAgglomeratedRidges(DenseMatrix & bettiNumbers,AgglomeratedTopology & topo, std::ostream & os);
};

#endif /* AGGLOMERATEDTOPOLOGYCHECK_HPP_ */
