-- phpMyAdmin SQL Dump
-- version 5.2.0
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Waktu pembuatan: 13 Des 2024 pada 15.23
-- Versi server: 10.4.24-MariaDB
-- Versi PHP: 8.1.6

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `skripsi_tes`
--

-- --------------------------------------------------------

--
-- Struktur dari tabel `alembic_version`
--

CREATE TABLE `alembic_version` (
  `version_num` varchar(32) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data untuk tabel `alembic_version`
--

INSERT INTO `alembic_version` (`version_num`) VALUES
('56ffa939fea8');

-- --------------------------------------------------------

--
-- Struktur dari tabel `categories`
--

CREATE TABLE `categories` (
  `id` int(11) NOT NULL,
  `name` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data untuk tabel `categories`
--

INSERT INTO `categories` (`id`, `name`) VALUES
(1, 'lada_halus'),
(2, 'lada_butir');

-- --------------------------------------------------------

--
-- Struktur dari tabel `sales`
--

CREATE TABLE `sales` (
  `id` int(11) NOT NULL,
  `date` varchar(7) NOT NULL,
  `total` float NOT NULL,
  `upload_time` datetime DEFAULT NULL,
  `category_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data untuk tabel `sales`
--

INSERT INTO `sales` (`id`, `date`, `total`, `upload_time`, `category_id`) VALUES
(1, '01-2020', 17, '2024-12-05 16:12:43', 1),
(2, '02-2020', 10, '2024-12-05 16:12:43', 1),
(3, '03-2020', 30, '2024-12-05 16:12:43', 1),
(4, '04-2020', 81, '2024-12-05 16:12:43', 1),
(5, '05-2020', 78, '2024-12-05 16:12:43', 1),
(6, '06-2020', 122, '2024-12-05 16:12:43', 1),
(7, '07-2020', 141, '2024-12-05 16:12:43', 1),
(8, '08-2020', 90, '2024-12-05 16:12:43', 1),
(9, '09-2020', 85, '2024-12-05 16:12:43', 1),
(10, '10-2020', 74, '2024-12-05 16:12:43', 1),
(11, '11-2020', 107, '2024-12-05 16:12:43', 1),
(12, '12-2020', 99, '2024-12-05 16:12:43', 1),
(13, '01-2021', 114, '2024-12-05 16:12:43', 1),
(14, '02-2021', 135, '2024-12-05 16:12:43', 1),
(15, '03-2021', 87, '2024-12-05 16:12:43', 1),
(16, '04-2021', 80, '2024-12-05 16:12:43', 1),
(17, '05-2021', 116, '2024-12-05 16:12:43', 1),
(18, '06-2021', 130, '2024-12-05 16:12:43', 1),
(19, '07-2021', 119, '2024-12-05 16:12:43', 1),
(20, '08-2021', 130, '2024-12-05 16:12:43', 1),
(21, '09-2021', 158, '2024-12-05 16:12:43', 1),
(22, '10-2021', 126, '2024-12-05 16:12:43', 1),
(23, '11-2021', 108, '2024-12-05 16:12:43', 1),
(24, '12-2021', 137, '2024-12-05 16:12:43', 1),
(25, '01-2022', 104, '2024-12-05 16:12:43', 1),
(26, '02-2022', 99, '2024-12-05 16:12:43', 1),
(27, '03-2022', 129, '2024-12-05 16:12:43', 1),
(28, '04-2022', 148, '2024-12-05 16:12:43', 1),
(29, '05-2022', 94, '2024-12-05 16:12:43', 1),
(30, '06-2022', 112, '2024-12-05 16:12:43', 1),
(31, '07-2022', 112, '2024-12-05 16:12:43', 1),
(32, '08-2022', 70, '2024-12-05 16:12:43', 1),
(33, '09-2022', 161, '2024-12-05 16:12:43', 1),
(34, '10-2022', 84, '2024-12-05 16:12:43', 1),
(35, '11-2022', 88, '2024-12-05 16:12:43', 1),
(36, '12-2022', 100, '2024-12-05 16:12:43', 1),
(37, '01-2023', 115, '2024-12-05 16:12:43', 1),
(38, '02-2023', 96, '2024-12-05 16:12:43', 1),
(39, '03-2023', 155, '2024-12-05 16:12:43', 1),
(40, '04-2023', 101, '2024-12-05 16:12:43', 1),
(41, '05-2023', 209, '2024-12-05 16:12:43', 1),
(42, '06-2023', 164, '2024-12-05 16:12:43', 1),
(43, '07-2023', 158, '2024-12-05 16:12:43', 1),
(44, '08-2023', 202, '2024-12-05 16:12:43', 1),
(45, '09-2023', 130, '2024-12-05 16:12:43', 1),
(46, '10-2023', 205, '2024-12-05 16:12:43', 1),
(47, '11-2023', 178, '2024-12-05 16:12:43', 1),
(48, '12-2023', 195, '2024-12-05 16:12:43', 1),
(49, '01-2024', 132, '2024-12-05 16:12:43', 1),
(50, '02-2024', 196, '2024-12-05 16:12:43', 1),
(51, '03-2024', 190, '2024-12-05 16:12:43', 1),
(52, '04-2024', 189, '2024-12-05 16:12:43', 1),
(53, '05-2024', 211, '2024-12-05 16:12:43', 1),
(54, '06-2024', 202, '2024-12-05 16:12:43', 1),
(55, '07-2024', 185, '2024-12-05 16:12:43', 1),
(58, '05-2020', 20, '2024-11-12 15:14:37', 2),
(59, '06-2020', 23, '2024-11-12 15:14:37', 2),
(60, '07-2020', 18, '2024-11-12 15:14:37', 2),
(61, '08-2020', 30, '2024-11-12 15:14:37', 2),
(62, '09-2020', 28, '2024-11-12 15:14:37', 2),
(63, '10-2020', 45, '2024-11-12 15:14:37', 2),
(64, '11-2020', 26, '2024-11-12 15:14:37', 2),
(65, '12-2020', 25, '2024-11-12 15:14:37', 2),
(66, '01-2021', 22, '2024-11-12 15:14:37', 2),
(67, '02-2021', 24, '2024-11-12 15:14:37', 2),
(68, '03-2021', 64, '2024-11-12 15:14:37', 2),
(69, '04-2021', 30, '2024-11-12 15:14:37', 2),
(70, '05-2021', 39, '2024-11-12 15:14:37', 2),
(71, '06-2021', 22, '2024-11-12 15:14:37', 2),
(72, '07-2021', 50, '2024-11-12 15:14:37', 2),
(73, '08-2021', 16, '2024-11-12 15:14:37', 2),
(74, '09-2021', 15, '2024-11-12 15:14:37', 2),
(75, '10-2021', 19, '2024-11-12 15:14:37', 2),
(76, '11-2021', 41, '2024-11-12 15:14:37', 2),
(77, '12-2021', 28, '2024-11-12 15:14:37', 2),
(78, '01-2022', 29, '2024-11-12 15:14:37', 2),
(79, '02-2022', 32, '2024-11-12 15:14:37', 2),
(80, '03-2022', 27, '2024-11-12 15:14:37', 2),
(81, '04-2022', 29, '2024-11-12 15:14:37', 2),
(82, '05-2022', 13, '2024-11-12 15:14:37', 2),
(83, '06-2022', 12, '2024-11-12 15:14:37', 2),
(84, '07-2022', 17, '2024-11-12 15:14:37', 2),
(85, '08-2022', 16, '2024-11-12 15:14:37', 2),
(86, '09-2022', 33, '2024-11-12 15:14:37', 2),
(87, '10-2022', 23, '2024-11-12 15:14:37', 2),
(88, '11-2022', 24, '2024-11-12 15:14:37', 2),
(89, '12-2022', 24, '2024-11-12 15:14:37', 2),
(90, '01-2023', 38, '2024-11-12 15:14:37', 2),
(91, '02-2023', 43, '2024-11-12 15:14:37', 2),
(92, '03-2023', 35, '2024-11-12 15:14:37', 2),
(93, '04-2023', 22, '2024-11-12 15:14:37', 2),
(94, '05-2023', 28, '2024-11-12 15:14:37', 2),
(95, '06-2023', 31, '2024-11-12 15:14:37', 2),
(96, '07-2023', 16, '2024-11-12 15:14:37', 2),
(97, '08-2023', 24, '2024-11-12 15:14:37', 2),
(98, '09-2023', 22, '2024-11-12 15:14:37', 2),
(99, '10-2023', 18, '2024-11-12 15:14:37', 2),
(100, '11-2023', 30, '2024-11-12 15:14:37', 2),
(101, '12-2023', 25, '2024-11-12 15:14:37', 2),
(102, '01-2024', 22, '2024-11-12 15:14:37', 2),
(103, '02-2024', 20, '2024-11-12 15:14:37', 2),
(104, '03-2024', 26, '2024-11-12 15:14:37', 2),
(105, '04-2024', 23, '2024-11-12 15:14:37', 2),
(106, '05-2024', 39, '2024-11-12 15:14:37', 2),
(107, '06-2024', 31, '2024-11-12 15:14:37', 2),
(108, '07-2024', 28, '2024-11-12 15:14:37', 2),
(109, '08-2024', 70, '2024-11-12 15:14:37', 2),
(111, '08-2024', 175, '2024-12-05 16:12:43', 1),
(139, '09-2024', 303, '2024-12-05 16:12:43', 1);

-- --------------------------------------------------------

--
-- Struktur dari tabel `user`
--

CREATE TABLE `user` (
  `id` int(11) NOT NULL,
  `username` varchar(80) NOT NULL,
  `password` varchar(200) NOT NULL,
  `email` varchar(120) NOT NULL,
  `is_approved` tinyint(1) DEFAULT NULL,
  `role` varchar(20) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data untuk tabel `user`
--

INSERT INTO `user` (`id`, `username`, `password`, `email`, `is_approved`, `role`) VALUES
(1, 'mike12', '$2b$12$ZpmKGkfe7plOtfLPPZbvzeU.UOtGE5Pj2rPiYQewQvCnwHIOEHAtO', 'yesayamichael16@gmail.com', NULL, NULL),
(2, 'admin', '$2b$12$hS0F9YrFg6S8KJeYL/6o0OT/W4VdjwXxO4.5iaOq4DdMRMhjq81a.', 'yesayamichael13@gmail.com', 1, 'admin'),
(3, 'mike1223', '$2b$12$QYbTGjkTrgF9ZgPdedbTs.rjbxvSJBB1MG8a41EpCz.FqPejZR1WK', 'yesayamichael15@gmail.com', 1, 'user'),
(4, 'mike1234', '$2b$12$NBGJJtNuzSXsXPQ/jrf6DOc0odqqqkdknTwkEkhhue4AlM8bCxvIW', 'yesayamichael14@gmail.com', 1, 'user'),
(5, 'mike12564', '$2b$12$.FulCGm7X8dgkPyA5U2fS.WY9BURs2CuD8JUtP03b4vwc2YgrL8Ui', 'yesayamichael17@gmail.com', 1, 'admin'),
(7, 'mike1212313', '$2b$12$.OYHBGHVb0SI7gux3lW8BOaAWlRdtJVFF4Cr8zJPxADTeodlMqk86', 'yesayamichael202@gmail.com', 1, 'user'),
(8, 'admin123123', '$2b$12$jk7bo07vLAlLCURMs6cgHukNEGxy/lwMQqnZHPt//zLth5PSvUKhO', 'yesayamichael55@gmail.com', 1, 'admin'),
(9, 'mike121221331', '$2b$12$myKONX3Cn..51D0L6pK9aOXpzKp4ats8r7cyI9uCrc/kLSY860vt6', 'yesayamichael76@gmail.com', 1, 'admin'),
(13, 'mik', '$2b$12$Je4peCkGtJP61N90cZeSxeKt1danNWkDNNTy5IJsvMJyrqn2K9aCq', 'yesayamichael2002@gmail.com', 1, 'user'),
(14, 'mike908', '$2b$12$J5syGj6rbPzAEHGbIWGSA.YoJIc7sdATx9q8b/ZRSn7eYlpJKt41W', 'yesayamichael213@gmail.com', 1, 'user'),
(19, 'tes123', '$2b$12$iwgQo3g9FjUkmJSv7KI0SeslRncqGBK4WcAaQOYs3yntPQ1/tdrAS', 'yesayamichael765@gmail.com', 1, 'user'),
(21, 'tes12333', '$2b$12$c7FvvniLSDgdbKA193JuGuKFUEJmSVclTKFAC8z1o5IwJDnNRYyGO', 'yesayamichael1612@gmail.com', 1, 'user'),
(24, 'mike9012', '$2b$12$wlq6ncG640nGNHdkyXTHeewnVXpWSTMgAjD1p9JpjmUkzABzFISXi', 'yesayamichael13213@gmail.com', 1, 'admin'),
(28, 'tes72', '$2b$12$GGX6b0BubTStgw67F7EHcuk62/.7kb/6gxJKrd/iVaJFp3Xqejc.y', 'tes@gmail.com', 1, 'user'),
(31, 'tes32', '$2b$12$xUXANANnyUwGjR4qJHI3HuhMxtwMX6wviuPHM0YvlfVc3Qz8iPJh2', 'tes32@gmail.com', 1, 'user'),
(34, 'tes4', '$2b$12$UwE09iq7YxvE1yrENEG8vu1Fu30XCfHaEWaHN90dJEXbJdHV7ioNC', 'yesayamichael1673@gmail.com', 1, 'user');

--
-- Indexes for dumped tables
--

--
-- Indeks untuk tabel `alembic_version`
--
ALTER TABLE `alembic_version`
  ADD PRIMARY KEY (`version_num`);

--
-- Indeks untuk tabel `categories`
--
ALTER TABLE `categories`
  ADD PRIMARY KEY (`id`);

--
-- Indeks untuk tabel `sales`
--
ALTER TABLE `sales`
  ADD PRIMARY KEY (`id`),
  ADD KEY `fk_category` (`category_id`);

--
-- Indeks untuk tabel `user`
--
ALTER TABLE `user`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `username` (`username`),
  ADD UNIQUE KEY `email` (`email`);

--
-- AUTO_INCREMENT untuk tabel yang dibuang
--

--
-- AUTO_INCREMENT untuk tabel `categories`
--
ALTER TABLE `categories`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT untuk tabel `sales`
--
ALTER TABLE `sales`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=148;

--
-- AUTO_INCREMENT untuk tabel `user`
--
ALTER TABLE `user`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=35;

--
-- Ketidakleluasaan untuk tabel pelimpahan (Dumped Tables)
--

--
-- Ketidakleluasaan untuk tabel `sales`
--
ALTER TABLE `sales`
  ADD CONSTRAINT `fk_category` FOREIGN KEY (`category_id`) REFERENCES `categories` (`id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
