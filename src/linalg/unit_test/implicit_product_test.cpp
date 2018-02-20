/*
  Copyright (c) 2018, Lawrence Livermore National Security, LLC. Produced at the
  Lawrence Livermore National Laboratory. LLNL-CODE-745557. All Rights reserved.
  See file COPYRIGHT for details.

  This file is part of the ParElag library. For more information and source code
  availability see http://github.com/LLNL/parelag.

  ParElag is free software; you can redistribute it and/or modify it under the
  terms of the GNU Lesser General Public License (as published by the Free
  Software Foundation) version 2.1 dated February 1999.
*/

#include "../ParELAG_ImplicitProductOperator.hpp"

int main(int argc, char **argv)
{
    std::cout << '\n' << std::string(50,'*') << '\n';
    spmat_ptr D1Sp{spmat_ptr(),D1};
    spmat_ptr D1T{mfem::Transpose(*D1)};

    std::cout << "size(D1Sp) = " << D1Sp->Height() << "x"
              << D1Sp->Width() << "\n"
              << "size(D1T) = " << D1T->Height() << "x"
              << D1T->Width() << "\n\n";

    auto testA = make_unique<ImplicitProductOperator>(M2);
    std::cout << "size(testA) = " << testA->Height() << "x"
              << testA->Width() << "\n"
              << "size(M2)    = " << M2->Height() << "x"
              << M2->Width() << "\n\n";

    testA->PreMultiply(D1T);
    std::cout << "size(testA)  = " << testA->Height() << "x"
              << testA->Width() << "\n"
              << "size(D1T*M2) = " << D1T->Height() << "x"
              << M2->Width() << "\n\n";

    testA->PostMultiply(D1Sp);
    std::cout << "size(testA) = " << testA->Height() << "x"
              << testA->Width() << "\n"
              << "size(spA)   = " << spA->Height() << "x"
              << spA->Width() << "\n";

    PARELAG_ASSERT(testA->Width() == spA->Width());
    mfem::Vector TestX(spA->Width()),
        TestB1(spA->Height()), TestB2(testA->Height());
    TestX.Randomize();
    TestB1 = 0.0; TestB2 = 0.0;

    spA->Mult(TestX,TestB1);
    testA->Mult(TestX,TestB2);

    std::cout << "||B1|| = " << TestB1.Norml2() << '\n'
              << "||B2|| = " << TestB2.Norml2() << '\n';

    TestX.Randomize(13);
    spA->MultTranspose(TestX,TestB1);
    testA->MultTranspose(TestX,TestB2);

    std::cout << "||B1|| = " << TestB1.Norml2() << '\n'
              << "||B2|| = " << TestB2.Norml2() << '\n';

    std::cout << std::string(50,'*') << "\n\n";
}
