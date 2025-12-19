// ============================================================================
// LEVEL 2: WHISPER CAPTURE - Low Volume Speech (Calibration)
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

#ifdef __APPLE__
#include <IOKit/serial/ioss.h>
#endif

std::atomic<bool> running(true);
void signal_handler(int signum) { std::cout << "\nInterrupt. Stopping...\n"; running = false; }

const int BAUD_RATE = 230400;
const std::string OUTPUT_FILE = "whisper_data.csv";
const int CYCLES = 10;

struct Phase { std::string label; int duration; std::string instruction; };
const std::vector<Phase> PHASES = {
    {"GHOST", 3, "WHISPER 'GHOST' - Low volume, breathy voice"},
    {"LEFT",  3, "WHISPER 'LEFT' - Low volume, breathy voice"},
    {"STOP",  3, "WHISPER 'STOP' - Low volume, breathy voice"},
    {"REST",  3, "Relax completely. Stay quiet."}
};

int open_serial_port(const std::string& port_name) {
    int fd = open(port_name.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd == -1) { perror("open_port"); return -1; }
    struct termios options;
    tcgetattr(fd, &options);
    cfsetispeed(&options, B230400); cfsetospeed(&options, B230400);
    options.c_cflag &= ~PARENB; options.c_cflag &= ~CSTOPB; options.c_cflag &= ~CSIZE; options.c_cflag |= CS8;
    options.c_cflag &= ~CRTSCTS;
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_iflag &= ~(IXON | IXOFF | IXANY); options.c_oflag &= ~OPOST;
    options.c_cc[VMIN] = 0; options.c_cc[VTIME] = 1;
    tcsetattr(fd, TCSANOW, &options); fcntl(fd, F_SETFL, 0);
    return fd;
}

std::vector<std::string> list_ports() {
    std::vector<std::string> ports; DIR* dir; struct dirent* ent;
    if ((dir = opendir("/dev")) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string name = ent->d_name;
            if (name.find("cu.usbserial") != std::string::npos || name.find("cu.usbmodem") != std::string::npos)
                ports.push_back("/dev/" + name);
        }
        closedir(dir);
    }
    return ports;
}

int main() {
    signal(SIGINT, signal_handler);
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  LEVEL 2: WHISPER (Low Volume Speech)                        ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    auto ports = list_ports();
    if (ports.empty()) { std::cerr << "No serial ports found!\n"; return 1; }
    std::cout << "Available ports:\n";
    for (size_t i = 0; i < ports.size(); ++i) std::cout << i << ": " << ports[i] << "\n";
    int selection = 0;
    if (ports.size() > 1) { std::cout << "Select port: "; std::cin >> selection; }
    else std::cout << "Auto-selecting " << ports[0] << "\n";
    if (selection < 0 || selection >= ports.size()) { std::cerr << "Invalid.\n"; return 1; }

    int serial_fd = open_serial_port(ports[selection]);
    if (serial_fd == -1) return 1;

    std::ofstream csv_file(OUTPUT_FILE, std::ios::app);
    csv_file.seekp(0, std::ios::end);
    if (csv_file.tellp() == 0) csv_file << "Label,Timestamp,RawValue\n";

    std::cout << "\nCalibrating..."; std::cout.flush();
    auto start_cal = std::chrono::steady_clock::now(); char buffer[1024];
    while (running && std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_cal).count() < 2) {
        read(serial_fd, buffer, sizeof(buffer)); std::cout << "."; std::cout.flush();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << " Done.\n\n>>> WHISPER SOFTLY! <<<\n\nStarting in 3..."; std::cout.flush();
    std::this_thread::sleep_for(std::chrono::seconds(1)); std::cout << "2..."; std::cout.flush();
    std::this_thread::sleep_for(std::chrono::seconds(1)); std::cout << "1...\n"; std::cout.flush();
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::string leftover = "";
    for (int cycle = 1; cycle <= CYCLES && running; ++cycle) {
        std::cout << "\n=== CYCLE " << cycle << "/" << CYCLES << " ===\n";
        for (const auto& phase : PHASES) {
            if (!running) break;
            std::cout << "\n>>> " << phase.label << " <<< : " << phase.instruction << "\n";
            auto phase_start = std::chrono::steady_clock::now(); int last_countdown = phase.duration;
            while (running) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - phase_start).count() / 1000.0;
                if (elapsed >= phase.duration) break;
                int time_left = phase.duration - (int)elapsed;
                if (time_left < last_countdown) { std::cout << (time_left + 1) << "... "; std::cout.flush(); last_countdown = time_left; }
                int n = read(serial_fd, buffer, sizeof(buffer) - 1);
                if (n > 0) {
                    buffer[n] = '\0'; leftover += buffer;
                    size_t pos;
                    while ((pos = leftover.find('\n')) != std::string::npos) {
                        std::string line = leftover.substr(0, pos); leftover.erase(0, pos + 1);
                        if (!line.empty() && line.back() == '\r') line.pop_back();
                        size_t commaPos = line.find(',');
                        if (commaPos != std::string::npos) {
                            std::string ts = line.substr(0, commaPos), val = line.substr(commaPos + 1);
                            bool valid = !ts.empty() && !val.empty();
                            for (char c : ts) if (!isdigit(c)) valid = false;
                            for (char c : val) if (!isdigit(c) && c != '-') valid = false;
                            if (valid) csv_file << phase.label << "," << ts << "," << val << "\n";
                        }
                    }
                    static int wc = 0; if (++wc % 1000 == 0) csv_file.flush();
                }
            }
        }
    }
    std::cout << "\n✅ Whisper Complete! → " << OUTPUT_FILE << "\n";
    close(serial_fd); csv_file.close(); return 0;
}
