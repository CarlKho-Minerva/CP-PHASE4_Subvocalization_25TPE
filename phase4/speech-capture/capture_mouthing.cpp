// ============================================================================
// LEVEL 3: MOUTHING CAPTURE - Open-Mouth Silent Speech (TRAINING DATA)
// ============================================================================
// This is the SAME core capture code from phase3/v2-emg-muscle/capture_guided.cpp
// ONLY THE LABELS ARE CHANGED for speech spectrum capture.
// "Mouthing" = Open-mouth, exaggerated silent speech with maximal jaw excursion
// ============================================================================

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <cstring>
#include <dirent.h>
#include <algorithm>
#include <numeric>
#include <csignal>
#include <atomic>

// Mac specific for high baud rates
#ifdef __APPLE__
#include <IOKit/serial/ioss.h>
#endif

// Global flag for interrupt handling
std::atomic<bool> running(true);

void signal_handler(int signum) {
    std::cout << "\nInterrupt signal (" << signum << ") received. Stopping...\n";
    running = false;
}

// Configuration
const int BAUD_RATE = 230400;
const std::string OUTPUT_FILE = "mouthing_data.csv";  // Changed for mouthing
const int CYCLES = 10;  // 10 cycles × 4 words = 40 reps per word

// ============================================================================
// SPEECH LABELS FOR MOUTHING (Level 3)
// Based on: docs/assignment/DATA_COLLECTION_PROTOCOL.md
// Chosen words: Distinct tongue gymnastics, not semantic meaning
// ============================================================================
struct Phase {
    std::string label;
    int duration;
    std::string instruction;
};

const std::vector<Phase> PHASES = {
    {"GHOST", 3, "MOUTH 'GHOST' - Exaggerate the G slam (tongue→palate)"},
    {"LEFT",  3, "MOUTH 'LEFT' - Exaggerate the L touch (tongue→alveolar ridge)"},
    {"STOP",  3, "MOUTH 'STOP' - Exaggerate the ST plosive (jaw engagement)"},
    {"REST",  3, "Relax completely. Mouth closed, tongue flat."}
};

// Serial Port Helper Functions
int open_serial_port(const std::string& port_name) {
    int fd = open(port_name.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd == -1) {
        perror("open_port: Unable to open serial port");
        return -1;
    }

    struct termios options;
    tcgetattr(fd, &options);

    // Set Baud Rate (Standard 230400)
    cfsetispeed(&options, B230400);
    cfsetospeed(&options, B230400);

    // 8N1
    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;

    // No flow control
    options.c_cflag &= ~CRTSCTS;

    // Raw input
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_iflag &= ~(IXON | IXOFF | IXANY);
    options.c_oflag &= ~OPOST;

    // Blocking read with timeout (VMIN=0, VTIME=1 -> 0.1s timeout)
    options.c_cc[VMIN] = 0;
    options.c_cc[VTIME] = 1;

    tcsetattr(fd, TCSANOW, &options);

    // Ensure non-blocking flag is off for read()
    fcntl(fd, F_SETFL, 0);

    return fd;
}

std::vector<std::string> list_ports() {
    std::vector<std::string> ports;
    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir("/dev")) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string name = ent->d_name;
            if (name.find("cu.usbserial") != std::string::npos ||
                name.find("cu.usbmodem") != std::string::npos) {
                ports.push_back("/dev/" + name);
            }
        }
        closedir(dir);
    }
    return ports;
}

// Main Logic
int main() {
    // Register signal handler
    signal(SIGINT, signal_handler);

    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  LEVEL 3: MOUTHING CAPTURE (Training Data)                   ║\n";
    std::cout << "║  Open-mouth, exaggerate tongue movements                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    // 1. Select Port
    auto ports = list_ports();
    if (ports.empty()) {
        std::cerr << "No serial ports found!\n";
        return 1;
    }

    std::cout << "Available ports:\n";
    for (size_t i = 0; i < ports.size(); ++i) {
        std::cout << i << ": " << ports[i] << "\n";
    }

    int selection = 0;
    if (ports.size() > 1) {
        std::cout << "Select port number: ";
        std::cin >> selection;
    } else {
        std::cout << "Auto-selecting " << ports[0] << "\n";
    }

    if (selection < 0 || selection >= ports.size()) {
        std::cerr << "Invalid selection.\n";
        return 1;
    }

    std::string port_name = ports[selection];
    std::cout << "Connecting to " << port_name << " at " << BAUD_RATE << " baud...\n";

    int serial_fd = open_serial_port(port_name);
    if (serial_fd == -1) return 1;

    // 2. Open CSV
    std::ofstream csv_file(OUTPUT_FILE, std::ios::app);
    if (!csv_file.is_open()) {
        std::cerr << "Could not open CSV file.\n";
        close(serial_fd);
        return 1;
    }

    // Check if empty to write header
    csv_file.seekp(0, std::ios::end);
    if (csv_file.tellp() == 0) {
        csv_file << "Label,Timestamp,RawValue\n";
    }

    // 3. Calibration (Visual only)
    std::cout << "\n--- CALIBRATION ---\n";
    std::cout << "Keep face RELAXED, mouth closed for 2 seconds...";
    std::cout.flush();

    auto start_cal = std::chrono::steady_clock::now();
    char buffer[1024];
    while (running && std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_cal).count() < 2) {
        int n = read(serial_fd, buffer, sizeof(buffer));
        if (n > 0) {
            std::cout << ".";
            std::cout.flush();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << "\nCalibration Done.\n";

    if (!running) {
        close(serial_fd);
        csv_file.close();
        return 0;
    }

    // 4. Main Loop
    std::cout << "\n>>> REMEMBER: EXAGGERATE your mouth movements! <<<\n";
    std::cout << ">>> This is TRAINING data - big signals are good! <<<\n\n";
    std::cout << "Starting Capture in 3 seconds...\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "2...\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "1...\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::string leftover = "";

    for (int cycle = 1; cycle <= CYCLES && running; ++cycle) {
        std::cout << "\n=== CYCLE " << cycle << "/" << CYCLES << " ===\n";

        for (const auto& phase : PHASES) {
            if (!running) break;
            std::cout << "\n>>> " << phase.label << " <<< : " << phase.instruction << "\n";

            auto phase_start = std::chrono::steady_clock::now();
            int last_countdown = phase.duration;

            while (running) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - phase_start).count() / 1000.0;

                if (elapsed >= phase.duration) break;

                // Countdown
                int time_left = phase.duration - (int)elapsed;
                if (time_left < last_countdown) {
                    std::cout << (time_left + 1) << "... ";
                    std::cout.flush();
                    last_countdown = time_left;
                }

                // Read Data
                int n = read(serial_fd, buffer, sizeof(buffer) - 1);
                if (n > 0) {
                    buffer[n] = '\0';
                    leftover += buffer;

                    size_t pos;
                    while ((pos = leftover.find('\n')) != std::string::npos) {
                        std::string line = leftover.substr(0, pos);
                        leftover.erase(0, pos + 1);

                        // Remove \r if present
                        if (!line.empty() && line.back() == '\r') {
                            line.pop_back();
                        }

                        // Parse: Timestamp,RawValue
                        size_t commaPos = line.find(',');
                        if (commaPos != std::string::npos) {
                            std::string ts_str = line.substr(0, commaPos);
                            std::string val_str = line.substr(commaPos + 1);

                            // Basic validation: ensure not empty and looks numeric
                            bool valid = !ts_str.empty() && !val_str.empty();
                            for (char c : ts_str) if (!isdigit(c)) valid = false;
                            for (char c : val_str) if (!isdigit(c) && c != '-') valid = false;

                            if (valid) {
                                // Write to CSV
                                csv_file << phase.label << "," << ts_str << "," << val_str << "\n";
                            }
                        }
                    }
                    // Periodic flush
                    static int write_count = 0;
                    if (++write_count % 1000 == 0) csv_file.flush();
                }
            }
        }
    }

    std::cout << "\n✅ Mouthing Capture Complete! Data saved to " << OUTPUT_FILE << "\n";
    std::cout << "Next step: Run ./capture_subvocal for Level 4 testing data\n";
    close(serial_fd);
    csv_file.close();
    return 0;
}
