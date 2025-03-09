#include <iostream>
#include <vector>
#include <cmath>
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TCanvas.h"

const double pionMass = 0.13957; // GeV (assume all particles are pions)


double computeInvariantMass(const std::vector<float>& ppt,
                            const std::vector<float>& peta,
                            const std::vector<float>& pphi,
                            const std::vector<float>& pm) {
    TLorentzVector totalP4;

    for (size_t i = 0; i < ppt.size(); i++) {
        TLorentzVector p4;
	p4.SetPtEtaPhiM(ppt[i],peta[i],pphi[i],pm[i]);
        totalP4 += p4;
    }

    return totalP4.M(); // Returns the invariant mass
}

void analyse() {
    // Open the ROOT file
    TFile* file = TFile::Open("/eos/user/f/fdibello/BigBello/delphes/test.root");
    if (!file || file->IsZombie()) {
        std::cerr << "Error: Could not open file!" << std::endl;
        return;
    }

    // Get the TTree
    TTree* tree = (TTree*)file->Get("tree"); // Replace "TTreeName" with actual tree name

    // Define variables to store data
    std::vector<float>* part_px = nullptr;
    std::vector<float>* part_py = nullptr;
    std::vector<float>* part_pz = nullptr;
    std::vector<float>* part_pt = nullptr;
    std::vector<float>* part_eta = nullptr;
    std::vector<float>* part_phi = nullptr;
    std::vector<float>* part_mass = nullptr;
    std::vector<float>* part_pid = nullptr;
    std::vector<float>* part_charge = nullptr;
    std::vector<int>* part_isFromD = nullptr;
    std::vector<int>* part_isFromDStar = nullptr;
    std::vector<float>* part_massReco = nullptr;

    // Set branch addresses
    tree->SetBranchAddress("part_px", &part_px);
    tree->SetBranchAddress("part_py", &part_py);
    tree->SetBranchAddress("part_pz", &part_pz);
    tree->SetBranchAddress("part_pt", &part_pt);
    tree->SetBranchAddress("part_eta", &part_eta);
    tree->SetBranchAddress("part_phi", &part_phi);
    tree->SetBranchAddress("part_mass", &part_mass);
    tree->SetBranchAddress("part_pid", &part_pid);
    tree->SetBranchAddress("part_charge", &part_charge);
    tree->SetBranchAddress("part_isFromD", &part_isFromD);
    tree->SetBranchAddress("part_isFromDStar", &part_isFromDStar);
    tree->SetBranchAddress("part_massReco", &part_massReco);

    // Create histogram for ΔM = M(D*) - M(D)
    TH1F* h_massDiff = new TH1F("h_massDiff", "Invariant Mass Difference (D* - D); #Delta M (GeV); Events", 
                                100, 0., 0.3); // 50 bins, range [0.1, 0.2] GeV

    // Loop over all events
    Long64_t nEntries = tree->GetEntries();
    cout<<" entries "<<nEntries<<endl;
    for (Long64_t i = 0; i < nEntries; i++) {
        tree->GetEntry(i);
        std::vector<float> px_D, py_D, pz_D, pm_D,pid_D;
        std::vector<float> ppt_D,peta_D, pphi_D, pmass_D;
        std::vector<float> ppt_DStar,peta_DStar, pphi_DStar, pmass_DStar;
        std::vector<float> px_DStar, py_DStar, pz_DStar, pm_DStar;
        bool hasK = 0;
        int has2pi = 0;
        int D0k = 0;
        int D0pi = 0;
        // Loop over all particles in the event
//        std::cout<<" -------------------------------------- "<<std::endl;
        for (size_t j = 0; j < part_px->size(); j++) {
            if (part_isFromD->at(j) > 0 && part_charge->at(j) != 0) {  // Particle from D decay
                px_D.push_back(part_px->at(j));
                py_D.push_back(part_py->at(j));
                pz_D.push_back(part_pz->at(j));
                peta_D.push_back(part_eta->at(j));
                pphi_D.push_back(part_phi->at(j));
                pmass_D.push_back(part_mass->at(j));
                ppt_D.push_back(part_pt->at(j));
                pid_D.push_back(part_pid->at(j));
		if(abs(part_pid->at(j)) == 321) pm_D.push_back(0.500);
                else pm_D.push_back(0.140);
            }
            if ((part_isFromDStar->at(j) > 0 || part_isFromD->at(j) > 0 ) && part_charge->at(j) != 0) {  // Particle from D* decay

		px_DStar.push_back(part_px->at(j));
                py_DStar.push_back(part_py->at(j));
                pz_DStar.push_back(part_pz->at(j));
                peta_DStar.push_back(part_eta->at(j));
                pphi_DStar.push_back(part_phi->at(j));
                pmass_DStar.push_back(part_mass->at(j));
                ppt_DStar.push_back(part_pt->at(j));
		if(abs(part_pid->at(j)) == 321) pm_DStar.push_back(0.500);
                else pm_DStar.push_back(0.140);
//		std::cout<<" added particle "<<part_pid->at(j)<<std::endl;
		if(abs(part_pid->at(j)) == 321) hasK = true;
		if(abs(part_pid->at(j)) == 211) has2pi++;
            }
        }


        double massD = computeInvariantMass(ppt_D, peta_D, pphi_D,pmass_D);
        double massDStar = computeInvariantMass(ppt_DStar, peta_DStar, pphi_DStar,pmass_DStar);
        double deltaM = massDStar - massD;

        // Fill histogram
	if( pz_DStar.size() - pz_D.size() > 0 && fabs(massD-1.865) < 0.4 && pz_DStar.size() == 3 ){ 
	//if(deltaM!=-1 && fabs(massD-1.865) < 1.15 && hasK && has2pi == 2 && pz_DStar.size()-pz_D.size() == 1 && pz_DStar.size() == 3){ 
		
	h_massDiff->Fill(deltaM);

        // Print results
        std::cout << "Event " << i << ": M(D) = " << massD << " with particles in vertex D "<<px_D.size()
                  << " GeV, M(D*) = " << massDStar << " with particles in vertex D* "<<px_DStar.size()
                  << " GeV, ΔM = " << deltaM << " GeV" << std::endl;
	}
    }

    // Draw and save the histogram
    TCanvas* canvas = new TCanvas("canvas", "Mass Difference Histogram", 800, 600);
    h_massDiff->Draw();
    canvas->SaveAs("mass_difference_all.pdf"); // Save plot as PDF

    // Save histogram to file
    TFile* outFile = new TFile("output_hist.root", "RECREATE");
    h_massDiff->Write();
    outFile->Close();

    std::cout << "Histogram saved to mass_difference.pdf and output_hist.root" << std::endl;

    // Cleanup
    delete h_massDiff;
    delete canvas;
    file->Close();
}

