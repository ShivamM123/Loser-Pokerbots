#include <skeleton/actions.h>
#include <skeleton/constants.h>
#include <skeleton/runner.h>
#include <skeleton/states.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <array>
#include <map>

using namespace pokerbots::skeleton;

// --- TIER 1: REAL-TIME OPPONENT PROFILER ---
struct OpponentProfile {
    int totalHands = 0;
    int turnRiverAggressionEvents = 0;
    
    // Calculates the opponent's true frequency of late-street aggression
    double getLateStreetAggression() {
        return (double)turnRiverAggressionEvents / (totalHands + 1);
    }
};

struct Bot {
    std::mt19937 rng;
    OpponentProfile profile;

    Bot() : rng(std::chrono::steady_clock::now().time_since_epoch().count()) {}

    int rankValue(char rank) {
        static const std::map<char, int> v = {{'2',2},{'3',3},{'4',4},{'5',5},{'6',6},{'7',7},{'8',8},{'9',9},{'T',10},{'J',11},{'Q',12},{'K',13},{'A',14}};
        return v.at(rank);
    }

    void handleNewRound(GameInfoPtr gameState, RoundStatePtr roundState, int active) {
        profile.totalHands++;
    }

    void handleRoundOver(GameInfoPtr gameState, TerminalStatePtr terminalState, int active) {}

    // --- TIER 1: BAYESIAN NODE-LOCKED SOLVER ---
    double solveNodeLocked(const std::array<std::string, 2>& myCards, const std::vector<std::string>& boardCards, int iterations, char bounty, bool opponentBetting, int street) {
        if (iterations <= 0) return 0.5;
        
        std::vector<int> deck;
        for (int i = 2; i <= 14; ++i) for (int j = 0; j < 4; ++j) deck.push_back(i);
        
        auto removeCard = [&](const std::string& c) {
            if (c.empty()) return;
            auto it = std::find(deck.begin(), deck.end(), rankValue(c[0]));
            if (it != deck.end()) deck.erase(it);
        };
        for (const auto& c : myCards) removeCard(c);
        for (const auto& c : boardCards) removeCard(c);

        double total_ev = 0;
        int valid_iters = 0;
        int b_val = rankValue(bounty);
        
        // Dynamic Range Evaluation
        double aggro = profile.getLateStreetAggression();
        bool isNit = (aggro < 0.15 && profile.totalHands > 15);

        for (int i = 0; i < iterations; ++i) {
            std::shuffle(deck.begin(), deck.end(), rng);
            int opp_c1 = deck[0];
            int opp_c2 = deck[1];

            // REJECTION SAMPLING (Node-Locking)
            // If a known tight player bets the Turn/River, mathematically delete weak hands from their simulated range.
            if (opponentBetting && street >= 4 && isNit) {
                int boardMax = 0;
                for(const auto& b : boardCards) boardMax = std::max(boardMax, rankValue(b[0]));
                
                // Reject simulation branches where the opponent doesn't hold broadway, a pair, or the highest board card
                if (opp_c1 < 10 && opp_c2 < 10 && opp_c1 != opp_c2 && opp_c1 != boardMax && opp_c2 != boardMax) {
                    continue; 
                }
            }

            int opp_val = opp_c1 + opp_c2;
            int my_val = rankValue(myCards[0][0]) + rankValue(myCards[1][0]);
            bool bounty_hit = (rankValue(myCards[0][0]) == b_val || rankValue(myCards[1][0]) == b_val);
            
            for (const auto& b : boardCards) {
                my_val += rankValue(b[0]); 
                opp_val += rankValue(b[0]);
                if (rankValue(b[0]) == b_val) bounty_hit = true;
            }

            double res = (my_val > opp_val) ? 1.0 : (my_val == opp_val ? 0.5 : 0.0);
            if (bounty_hit) res *= 1.5; // Bounty Reward Shift 
            
            total_ev += res;
            valid_iters++;
        }
        return valid_iters > 0 ? total_ev / (double)valid_iters : 0.0;
    }

    Action getAction(GameInfoPtr gameState, RoundStatePtr roundState, int active) {
        auto legalActions = roundState->legalActions();
        int street = roundState->street;
        auto myCards = roundState->hands[active];
        float clock = gameState->gameClock;
        char bounty = roundState->bounties[active];
        
        std::vector<std::string> board;
        for (int i = 0; i < street && i < roundState->deck.size(); ++i) {
            if (!roundState->deck[i].empty()) board.push_back(roundState->deck[i]);
        }

        int myPip = roundState->pips[active];
        int oppPip = roundState->pips[1 - active];
        int pot = (STARTING_STACK - roundState->stacks[active]) + (STARTING_STACK - roundState->stacks[1-active]);
        int cost = oppPip - myPip;

        // --- REAL-TIME LIVE TRACKING ---
        if (street >= 4 && cost > 10) {
            profile.turnRiverAggressionEvents++;
        }

        int iters = (clock < 5.0) ? 50 : (street == 0 ? 500 : (street == 5 ? 25000 : 10000));
        double equity = solveNodeLocked(myCards, board, iters, bounty, cost > 0, street);
        double potOdds = (double)cost / (pot + cost + 0.0001);

        // --- TIER 1: ACTION ABSTRACTION & MIXED STRATEGY ---
        if (cost > 0) {
            if (equity < potOdds) {
                if (legalActions.find(Action::Type::FOLD) != legalActions.end()) return {Action::Type::FOLD};
            }
            if (equity > 0.85 && legalActions.find(Action::Type::RAISE) != legalActions.end()) {
                auto bounds = roundState->raiseBounds();
                // Dynamic Sizing: Overbet the absolute nuts, standard 3-bet the strong hands
                int target = (equity > 0.92) ? pot * 1.5 : myPip + cost * 3;
                return {Action::Type::RAISE, std::min(bounds[1], std::max(bounds[0], target))};
            }
            return {Action::Type::CALL};
        } else {
            if (legalActions.find(Action::Type::RAISE) != legalActions.end()) {
                auto bounds = roundState->raiseBounds();
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                double mix = dist(rng);

                // Value Betting: Mix 100% pot and 60% pot to remain mathematically unreadable
                if (equity > 0.75) {
                    int bet = (mix > 0.5) ? pot : (int)(pot * 0.6);
                    return {Action::Type::RAISE, std::min(bounds[1], std::max(bounds[0], bet))};
                } 
                // Balanced Bluffing: 30% of the time, throw a 33% pot probe bet with middling equity
                else if (equity > 0.45 && equity < 0.60 && mix > 0.70) {
                    int bet = (int)(pot * 0.33);
                    return {Action::Type::RAISE, std::min(bounds[1], std::max(bounds[0], bet))};
                }
            }
            return (legalActions.find(Action::Type::CHECK) != legalActions.end()) ? Action(Action::Type::CHECK) : Action(Action::Type::CALL);
        }
    }
};

int main(int argc, char *argv[]) {
    auto [host, port] = parseArgs(argc, argv);
    runBot<Bot>(host, port);
    return 0;
}